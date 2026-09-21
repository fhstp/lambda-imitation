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
        priorities: Optional per-slot sampling priority of shape ``(size,)``,
            used by :func:`create_prioritised_sequence_sample` and written by
            :func:`update_priorities`.  ``None`` (the default) means uniform
            sampling, which is what every non-prioritised caller gets.
    """

    info: dict[str, jax.Array]
    sampling_ok: jax.Array
    pos: int
    size: int
    priorities: jax.Array | None = None


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


def create_sequence_sample(
    buffer_size: int,
    sampling_size: int,
    sequence_size: int,
    keys: list[str],
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

    base_indices = jnp.vstack((jnp.arange(sequence_size),) * sampling_size)

    def sample(buffer: Buffer, key: jax.Array) -> Tuple[BufferSample, jax.Array]:
        sampling_ok = _valid_window_mask(buffer.sampling_ok, sequence_size)
        probs = sampling_ok.astype(jnp.float32)
        probs = probs / probs.sum()
        indices = jax.random.choice(key, buffer_size, (sampling_size, 1), p=probs)
        # Wrap around the ring: a window may legitimately start near the end
        # (the validity check above already padded circularly).  Without the
        # modulo, out-of-bounds gathers clamp to the last slot under jit.
        sequence_indices = (indices + base_indices) % buffer_size

        return (
            BufferSample(
                this_info=jax.tree.map(
                    lambda arr: arr[sequence_indices],
                    {k: buffer.info[k] for k in keys},
                ),
                next_info={},
            ),
            sequence_indices,
        )

    return sample


def _valid_window_mask(sampling_ok: jax.Array, sequence_size: int) -> jax.Array:
    """Per-start-index mask: True where a whole ``sequence_size`` window is valid.

    Circular min-reduction over ``sampling_ok`` — the shared core of the uniform
    and prioritised sequence samplers, so both agree on which windows exist.
    """
    x = sampling_ok.astype(jnp.int32)
    padded = jnp.concatenate([x, x[:sequence_size]])
    return jax.lax.reduce_window(
        padded,
        init_value=jnp.iinfo(padded.dtype).max,
        computation=jax.lax.min,
        window_dimensions=(sequence_size + 1,),
        window_strides=(1,),
        padding="VALID",
    )


def trace_priority(ratios: jax.Array, floor: float = 1e-3) -> jax.Array:
    """Trace mass of each window: the geometric mean of its clipped ratios.

    ``exp(mean_k log(clip(rho_k, floor, 1)))`` over the window's time axis
    (the last one).  This is monotone in the raw product ``prod_k rho_k`` that
    V-trace / Retrace actually apply to the window's tail, so it ranks windows
    identically, but it cannot underflow: a strict product over a 50-step
    window is exactly 0 for any window containing a single cut step, and
    clipping without the log gives ``1e-3 ** 50 = 1e-150``.

    Args:
        ratios: ``(..., T)`` importance ratios, time last.
        floor: Lower clip on each ratio; also sets the score of a fully cut
            window (``= floor``), keeping priorities strictly positive so no
            window is ever unreachable.

    Returns:
        ``(...)`` priorities in ``[floor, 1]``.
    """
    clipped = jnp.clip(ratios, floor, 1.0)
    return jnp.exp(jnp.mean(jnp.log(clipped), axis=-1))


def update_priorities(
    buffer: Buffer, start_indices: jax.Array, values: jax.Array
) -> Buffer:
    """Write new priorities for the given window start slots.

    Args:
        buffer: Current buffer state.
        start_indices: ``(batch,)`` window start slots, as returned by the
            prioritised sampler (its ``indices[:, 0]``).
        values: ``(batch,)`` new priorities, strictly positive.

    Returns:
        A new ``Buffer``.  Duplicate indices resolve to one of the writes
        (XLA scatter semantics); priorities are a heuristic, so this is
        deliberately not disambiguated.
    """
    return buffer._replace(
        priorities=buffer.priorities.at[start_indices].set(values)
    )


def create_prioritised_sequence_sample(
    buffer_size: int,
    sampling_size: int,
    sequence_size: int,
    keys: list[str],
    alpha: float = 0.0,
) -> Callable[[Buffer, jax.Array], Tuple[BufferSample, jax.Array, jax.Array]]:
    """Build a sequence sampler that draws windows in proportion to priority.

    ``P(i) proportional to priority_i ** alpha`` over valid window starts
    (invalid ones get zero mass).  ``alpha = 0`` is exactly uniform, so this
    sampler is a drop-in replacement that reproduces
    :func:`create_sequence_sample`'s draws for the same key.

    Args:
        buffer_size: Total slots in the buffer.
        sampling_size: Sequences per batch.
        sequence_size: Length of each sampled window.
        keys: Buffer keys to gather.
        alpha: Prioritisation exponent (0 = uniform, 1 = proportional).

    Returns:
        ``sample(buffer, key) -> (BufferSample, sequence_indices, probs)`` where
        ``probs`` are the draw probabilities of the selected windows — the
        quantity the caller needs to form importance-sampling weights
        ``w_i = (N * P_i) ** -beta`` correcting the bias this sampling adds.
    """
    base_indices = jnp.vstack((jnp.arange(sequence_size),) * sampling_size)

    def sample(
        buffer: Buffer, key: jax.Array
    ) -> Tuple[BufferSample, jax.Array, jax.Array]:
        valid = _valid_window_mask(buffer.sampling_ok, sequence_size).astype(
            jnp.float32
        )
        if alpha == 0.0:
            scores = valid
        else:
            scores = valid * jnp.power(buffer.priorities, alpha)
        probs = scores / jnp.maximum(scores.sum(), 1e-12)
        indices = jax.random.choice(key, buffer_size, (sampling_size, 1), p=probs)
        sequence_indices = (indices + base_indices) % buffer_size

        return (
            BufferSample(
                this_info=jax.tree.map(
                    lambda arr: arr[sequence_indices],
                    {k: buffer.info[k] for k in keys},
                ),
                next_info={},
            ),
            sequence_indices,
            probs[indices[:, 0]],
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
        # Max priority for unvisited slots, so every window is tried at least
        # once before its priority is measured (standard PER optimism).
        priorities=jnp.ones((size,)),
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
            # A freshly written slot has never been evaluated: give it max
            # priority so it is not starved before it is measured once.
            priorities=(
                None
                if buffer.priorities is None
                else buffer.priorities.at[pos].set(1.0)
            ),
        )

    sample = create_sample(size, sampling_size, this_step_infos, next_step_infos)

    return buffer, BufferFunctions(add, sample)
