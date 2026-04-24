"""Circular replay buffer for off-policy reinforcement learning.

The buffer stores transitions as a fixed-size ring, indexed by string keys so
the stored data schema is fully configurable by the caller.  Sampling is
restricted to slots whose "next" observation is valid (either the slot has a
successor in the ring, or the episode terminated there), tracked via the
boolean ``sampling_ok`` mask.

Typical usage::

    buffer, fns = create_buffer(
        shapes={"obs": (4,), "act": (2,)},
        size=10_000,
        sampling_size=256,
        this_step_infos=["obs", "act"],
        next_step_infos=["obs"],
    )
    buffer = fns.add(buffer, {"obs": obs, "act": act}, terminated=False)
    sample, indices = fns.sample(buffer, jax.random.key(0))
"""

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    """Immutable snapshot of the replay buffer.

    Attributes:
        info: Dict mapping string keys to arrays of shape ``(size, *item_shape)``.
            All entries share the same leading ``size`` dimension.
        sampling_ok: Boolean mask of shape ``(size,)``.  Slot ``i`` is True when
            it is safe to sample: the "next" observation at slot ``(i+1) % size``
            is valid (either the episode continued, or slot ``i`` was terminal).
        pos: Write cursor.  The *next* call to ``add`` will write to slot
            ``pos % size``.  Increases monotonically (not wrapped).
        size: Maximum number of transitions the buffer can hold.
    """

    info: dict[str, jax.Array]
    sampling_ok: jax.Array
    pos: int
    size: int


class BufferSample(NamedTuple):
    """A batch of transitions drawn from the buffer.

    Attributes:
        this_info: Dict of arrays with shape ``(batch, *item_shape)`` for the
            sampled timestep ``t``.
        next_info: Dict of arrays with shape ``(batch, *item_shape)`` for the
            successor timestep ``t+1`` (wrapping circularly).
    """

    this_info: dict[str, jax.Array]
    next_info: dict[str, jax.Array]


class BufferFunctions(NamedTuple):
    """Pair of pure functions returned by :func:`create_buffer`.

    Attributes:
        add: ``(Buffer, infos, terminated) -> Buffer`` -- insert one transition.
        sample: ``(Buffer, key) -> (BufferSample, indices)`` -- draw a batch.
    """

    add: Callable
    sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]]


class SequenceSample(NamedTuple):
    """A batch of contiguous sequences drawn from the buffer.

    Attributes:
        info: Dict mapping string keys to arrays of shape
            ``(batch, T, *item_shape)``.  Each row is a contiguous run of T
            consecutive buffer slots (wrapping circularly).
        mask: Float32 array of shape ``(batch, T)``.  Entry ``[b, t]`` is 1.0
            for every timestep up to and including the first terminal in
            sequence ``b``, and 0.0 for all timesteps that follow the first
            terminal.  Use this to mask out loss contributions from
            post-episode junk data.
        start_index: Integer array of shape ``(batch,)`` giving the buffer
            slot at which each sequence starts.
    """

    info: dict[str, jax.Array]
    mask: jax.Array
    start_index: jax.Array


def create_sequence_sample(
    buffer_size: int,
    batch_size: int,
    burn_in: int,
    sequence_length: int,
    keys: list[str],
    *,
    terminal_key: str,
) -> Callable[[Buffer, jax.Array], SequenceSample]:
    """Build a sequence-sampling function for a fixed buffer layout.

    A sequence of length ``sequence_length`` is valid to sample only when all
    ``sequence_length`` consecutive slots (wrapping circularly) have
    ``sampling_ok=True``.  The probability of each start position is uniform
    over valid starts.

    Args:
        buffer_size: Total capacity of the buffer (number of slots).
        batch_size: Number of sequences to draw per call.
        sequence_length: Length ``T`` of each sampled sequence.
        keys: Buffer keys to gather into ``SequenceSample.info``.  Must
            include ``terminal_key``.
        terminal_key: Key in ``buffer.info`` whose values are 0/1 float32
            flags marking episode termination.  Used to compute the mask.

    Returns:
        A pure function ``sample(buffer, key) -> SequenceSample``.  Assumes
        the buffer holds at least ``sequence_length`` valid consecutive slots;
        behaviour is undefined (NaN probabilities) if no valid start exists.
    """
    B = burn_in
    T = sequence_length

    # Precompute offset array — used every call but never changes.
    offsets = jnp.arange(-B, T)  # (T,)

    def sample(buffer: Buffer, key: jax.Array) -> SequenceSample:
        # Valid-start mask: every slot in the T-window must be sampling_ok.
        all_slots = jnp.arange(buffer_size)
        window = (all_slots[:, None] + offsets[None, :]) % buffer_size  # (size, T)
        valid = jnp.all(buffer.sampling_ok[window], axis=-1)  # (size,) bool

        probs = valid.astype(jnp.float32)
        probs = probs / probs.sum()
        starts = jax.random.choice(key, buffer_size, (batch_size,), p=probs)  # (batch,)

        # Gather contiguous sequences for every requested key.
        seq_idx = (starts[:, None] + offsets[None, :]) % buffer_size  # (batch, T)
        gathered = {k: buffer.info[k][seq_idx] for k in keys}

        # Mask: 1 up to and including the first terminal, 0 afterwards.
        # cummax over the terminal flags detects the first terminal position;
        # shifting right by one makes the terminal timestep itself included.
        term = gathered[terminal_key][:, B:].astype(jnp.float32)  # (batch, T)
        cummax = jnp.maximum.accumulate(term, axis=1)  # (batch, T)
        shifted = jnp.concatenate(
            [jnp.zeros((batch_size, B + 1), dtype=jnp.float32), cummax[:, :-1]], axis=1
        )
        mask = 1.0 - shifted  # (batch, T)

        return SequenceSample(info=gathered, mask=mask, start_index=starts), seq_idx

    return sample


def create_sample(
    buffer_size: int,
    sampling_size: int,
    this_keys: list[str],
    next_keys: list[str],
) -> Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]]:
    """Build a sampling function for a fixed buffer layout.

    The returned function can be called independently of :func:`create_buffer`,
    which is useful when the buffer is constructed elsewhere (e.g. inside
    ``create_iqlearn``).

    Args:
        buffer_size: Total capacity of the buffer (number of slots).
        sampling_size: Number of transitions to draw per call.
        this_keys: Buffer keys to include in ``BufferSample.this_info``.
        next_keys: Buffer keys to include in ``BufferSample.next_info``.

    Returns:
        A pure function ``sample(buffer, key) -> (BufferSample, indices)`` that
        samples ``sampling_size`` transitions uniformly from valid slots (those
        where ``buffer.sampling_ok`` is True) using the provided PRNG key.
        Returns both the sample and the raw integer indices used, which callers
        may need for priority updates.
    """

    def sample(buffer: Buffer, key: jax.Array) -> Tuple[BufferSample, jax.Array]:
        probs = jnp.astype(buffer.sampling_ok, jnp.float32)
        probs = probs / probs.sum()
        indices = jax.random.choice(key, buffer_size, (sampling_size,), p=probs)

        return (
            BufferSample(
                this_info=jax.tree.map(
                    lambda arr: arr[indices], {k: buffer.info[k] for k in this_keys}
                ),
                next_info=jax.tree.map(
                    lambda arr: arr[(indices + 1) % buffer_size],
                    {k: buffer.info[k] for k in next_keys},
                ),
            ),
            indices,
        )

    return sample


def create_buffer(
    shapes: dict[str, tuple[int, ...]],
    size: int,
    sampling_size: int,
    this_step_infos: list[str],
    next_step_infos: list[str],
) -> Tuple[Buffer, BufferFunctions]:
    """Create an empty circular replay buffer and its associated functions.

    All data is stored as JAX arrays in a plain Python dict keyed by strings,
    so the schema is fully caller-defined.  The buffer is immutable: ``add``
    returns a new ``Buffer`` rather than mutating in place.

    ``sampling_ok`` slot rules (applied every ``add`` call):

    - The slot being written is marked False unless the episode terminates
      there (its "next" is unknown for non-terminal steps).
    - The previous slot ``(pos - 1) % size`` is marked True once the current
      slot provides a valid successor.  On the very first add this wraps to
      slot ``size - 1`` but leaves it False because no successor exists yet.
    - A terminal slot is marked True immediately: the episode ends there, so
      no valid "next" is needed for bootstrapping.

    These two sequential single-index writes are deliberately kept separate to
    avoid duplicate-index scatter semantics (which are undefined under
    ``jax.jit`` / XLA when the same index appears more than once).

    Args:
        shapes: Mapping from key name to item shape (without the leading batch
            dimension).  E.g. ``{"obs": (4,), "act": (2,), "rew": ()}``.
        size: Maximum number of transitions stored.  Older entries are
            overwritten once the buffer is full.
        sampling_size: Default batch size used by the returned ``sample``
            function.
        this_step_infos: Keys whose values at timestep *t* should appear in
            ``BufferSample.this_info``.
        next_step_infos: Keys whose values at timestep *t+1* should appear in
            ``BufferSample.next_info``.

    Returns:
        A ``(Buffer, BufferFunctions)`` pair.  The buffer is fully zeroed.
        ``BufferFunctions.add`` and ``BufferFunctions.sample`` are pure
        functions; pass the ``Buffer`` as the first argument on every call.
    """
    info = {k: jnp.zeros((size,) + shapes[k]) for k in shapes}
    buffer = Buffer(
        info,
        sampling_ok=jnp.zeros((size,), dtype=jnp.bool),
        pos=0,
        size=size,
    )

    def add(buffer: Buffer, infos: dict[str, jax.Array], terminated: bool) -> Buffer:
        """Insert one transition and return the updated buffer.

        Only keys present in ``infos`` are written; missing keys retain their
        current values.  The ``sampling_ok`` mask is updated via two sequential
        single-index writes (predecessor rule + terminal rule) to avoid
        duplicate-index scatter behaviour that is undefined under
        ``jax.jit`` / XLA.

        Args:
            buffer: Current buffer state.
            infos: Dict of arrays (one per key) with item shapes matching those
                passed to ``create_buffer``.  Keys absent from this dict are
                left unchanged at the current write slot.
            terminated: Whether this step ends the episode.  A terminal step is
                immediately marked sampleable because its "next" observation is
                not needed for bootstrapping.

        Returns:
            A new ``Buffer`` with the transition written and all counters updated.
        """
        pos = (buffer.pos) % buffer.size

        new_infos = {
            key: (
                buffer.info[key].at[pos].set(infos[key])
                if key in infos
                else buffer.info[key]
            )
            for key in buffer.info
        }

        prev = (pos - 1) % buffer.size
        # Mark the predecessor slot as having a valid successor.  On the very
        # first add (buffer.pos == 0) prev wraps to size-1 but has no real
        # successor yet, so we leave it False.
        sampling_ok = buffer.sampling_ok.at[prev].set(buffer.pos > 0)
        # Current slot: sampleable immediately only when the episode ends here
        # (no bootstrapped "next" is needed).  Otherwise clear it until a
        # future add writes to the following slot and marks it via the rule
        # above.  Two separate writes avoid duplicate-index scatter semantics
        # that are undefined under jax.jit / XLA.
        sampling_ok = sampling_ok.at[pos].set(terminated)

        return Buffer(
            new_infos,
            sampling_ok=sampling_ok,
            pos=buffer.pos + 1,
            size=buffer.size,
        )

    sample = create_sample(size, sampling_size, this_step_infos, next_step_infos)

    return buffer, BufferFunctions(add, sample)
