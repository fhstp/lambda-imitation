from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    info: dict[str, jax.Array]
    sampling_ok: jax.Array
    pos: int
    size: int


class BufferSample(NamedTuple):
    this_info: dict[str, jax.Array]
    next_info: dict[str, jax.Array]


class BufferFunctions(NamedTuple):
    add: Callable
    sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]]


def create_sample(buffer_size, sampling_size, this_keys, next_keys):
    def sample(buffer: Buffer, key):
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
    shapes: dict[str, tuple[int]],
    size: int,
    sampling_size: int,
    this_step_infos: list[str],
    next_step_infos: list[str],
) -> Tuple[Buffer, BufferFunctions]:
    info = {k: jnp.zeros((size,) + shapes[k]) for k in shapes}
    buffer = Buffer(
        info,
        sampling_ok=jnp.zeros((size,), dtype=jnp.bool),
        pos=0,
        size=size,
    )

    def add(buffer: Buffer, infos: dict[str, jax.Array], terminated: bool) -> Buffer:
        pos = (buffer.pos) % buffer.size

        new_infos = {
            key: (
                buffer.info[key].at[pos].set(infos[key])
                if key in infos
                else buffer.info[key]
            )
            for key in buffer.info
        }

        sampling_ok = buffer.sampling_ok.at[pos].set(False)
        sampling_ok = sampling_ok.at[
            jnp.array([buffer.size - 1, (pos - 1) % buffer.size, pos])
        ].set(
            (
                jnp.logical_and(buffer.pos >= buffer.size, sampling_ok[-1]),
                buffer.pos > 0,
                terminated,
            )
        )

        return Buffer(
            new_infos,
            sampling_ok=sampling_ok,
            pos=buffer.pos + 1,
            size=buffer.size,
        )

    sample = create_sample(size, sampling_size, this_step_infos, next_step_infos)

    return buffer, BufferFunctions(add, sample)

