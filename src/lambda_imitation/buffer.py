"""Immutable circular replay with uniform transition and sequence sampling."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    """Ring storage keyed by field name, with a monotonically increasing cursor.

    ``sampling_ok[i]`` is true when slot i has a valid successor or is terminal.
    Terminal transitions are immediately sampleable. A newly written transition
    makes its predecessor sampleable, except on the very first insertion.
    """

    info: dict[str, jax.Array]
    sampling_ok: jax.Array
    pos: int
    size: int


class BufferSample(NamedTuple):
    this_info: dict[str, jax.Array]
    next_info: dict[str, jax.Array]


class BufferFunctions(NamedTuple):
    add: Callable
    sample: Callable


def create_sample(buffer_size, sampling_size, this_keys, next_keys):
    """Return ``sample(buffer, key) -> (BufferSample, indices)``.

    Draw with replacement uniformly over valid slots. The caller must ensure
    there is at least one valid transition before sampling.
    """
    def sample(buffer, key):
        probs = buffer.sampling_ok.astype(jnp.float32)
        probs = probs / probs.sum()
        indices = jax.random.choice(key, buffer_size, (sampling_size,), p=probs)
        return BufferSample(
            {k: buffer.info[k][indices] for k in this_keys},
            {k: buffer.info[k][(indices + 1) % buffer_size] for k in next_keys},
        ), indices

    return sample


def _valid_window_mask(sampling_ok, sequence_size):
    """Require the window and its successor to be valid, including ring wrap.

    This is the sampling convention used by the reported experiments. The
    invalid newest nonterminal slot also prevents crossing the ring's write
    cursor into older data.
    """
    x = sampling_ok.astype(jnp.int32)
    return jax.lax.reduce_window(
        jnp.concatenate([x, x[:sequence_size]]), jnp.iinfo(x.dtype).max,
        jax.lax.min, (sequence_size + 1,), (1,), "VALID")


def create_sequence_sample(buffer_size, sampling_size, sequence_size, keys):
    """Uniform contiguous windows, returned in batch-major ``(B, T, ...)`` order.

    Windows may cross episode boundaries; the learner resets its recurrent
    state using stored termination flags. The caller must first collect enough
    transitions for at least one complete valid window.
    """
    if not 0 < sequence_size < buffer_size:
        raise ValueError("sequence_size must be positive and smaller than buffer_size")
    base_indices = jnp.vstack((jnp.arange(sequence_size),) * sampling_size)

    def sample(buffer, key):
        valid = _valid_window_mask(buffer.sampling_ok, sequence_size)
        probs = valid.astype(jnp.float32)
        probs = probs / probs.sum()
        starts = jax.random.choice(key, buffer_size, (sampling_size, 1), p=probs)
        indices = (starts + base_indices) % buffer_size
        return BufferSample({k: buffer.info[k][indices] for k in keys}, {}), indices

    return sample


def create_buffer(shapes, size, sampling_size, this_step_infos, next_step_infos):
    """Allocate zeroed replay and return its pure ``add`` and ``sample`` closures.

    ``shapes`` maps names to per-transition shapes. All fields are float32 JAX
    arrays; discrete actions are stored as scalar indices of shape ``(1,)``.
    """
    buffer = Buffer(
        {k: jnp.zeros((size, *shape), jnp.float32) for k, shape in shapes.items()},
        jnp.zeros((size,), dtype=jnp.bool_), 0, size)

    def add(buffer, infos, terminated):
        pos = buffer.pos % buffer.size
        info = {k: arr.at[pos].set(infos[k]) if k in infos else arr
                for k, arr in buffer.info.items()}
        # Separate scatter writes also handle a one-slot buffer correctly.
        valid = buffer.sampling_ok.at[(pos - 1) % buffer.size].set(buffer.pos > 0)
        valid = valid.at[pos].set(terminated)
        return Buffer(info, valid, buffer.pos + 1, buffer.size)

    return buffer, BufferFunctions(
        add, create_sample(size, sampling_size, this_step_infos, next_step_infos))
