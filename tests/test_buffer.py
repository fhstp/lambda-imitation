"""Tests for :mod:`lambda_imitation.buffer`."""

import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.buffer import (
    Buffer,
    BufferFunctions,
    BufferSample,
    create_buffer,
    create_sample,
    create_sequence_sample,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def shapes():
    return {"obs": (4,), "act": (2,), "rew": ()}


@pytest.fixture
def empty_buffer(shapes):
    buffer, fns = create_buffer(
        shapes=shapes,
        size=8,
        sampling_size=4,
        this_step_infos=["obs", "act", "rew"],
        next_step_infos=["obs"],
    )
    return buffer, fns


def _step(i: int, shapes: dict[str, tuple[int, ...]]) -> dict[str, jax.Array]:
    """Deterministic dummy transition keyed by step index."""
    return {
        "obs": jnp.full(shapes["obs"], float(i)),
        "act": jnp.full(shapes["act"], float(i) * 0.1),
        "rew": jnp.asarray(float(i) * 10.0),
    }


# ---------------------------------------------------------------------------
# create_buffer: structure
# ---------------------------------------------------------------------------


class TestCreateBuffer:
    def test_returns_buffer_and_functions(self, empty_buffer):
        buffer, fns = empty_buffer
        assert isinstance(buffer, Buffer)
        assert isinstance(fns, BufferFunctions)
        assert callable(fns.add)
        assert callable(fns.sample)

    def test_info_arrays_zeroed_with_correct_shape(self, shapes):
        buffer, _ = create_buffer(
            shapes=shapes,
            size=8,
            sampling_size=4,
            this_step_infos=["obs"],
            next_step_infos=["obs"],
        )
        for key, item_shape in shapes.items():
            arr = buffer.info[key]
            assert arr.shape == (8,) + item_shape
            assert jnp.all(arr == 0)

    def test_sampling_ok_all_false_initially(self, empty_buffer):
        buffer, _ = empty_buffer
        assert buffer.sampling_ok.shape == (8,)
        assert buffer.sampling_ok.dtype == jnp.bool_
        assert not jnp.any(buffer.sampling_ok)

    def test_pos_starts_at_zero(self, empty_buffer):
        buffer, _ = empty_buffer
        assert int(buffer.pos) == 0

    def test_size_field_matches_arg(self, empty_buffer):
        buffer, _ = empty_buffer
        assert buffer.size == 8


# ---------------------------------------------------------------------------
# add: writes and pos
# ---------------------------------------------------------------------------


class TestAdd:
    def test_writes_provided_keys_into_current_slot(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(1, shapes), terminated=False)
        assert jnp.all(buffer.info["obs"][0] == 1.0)
        assert jnp.allclose(buffer.info["act"][0], 0.1)
        assert float(buffer.info["rew"][0]) == 10.0

    def test_does_not_touch_other_slots(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(1, shapes), terminated=False)
        # all slots 1..7 still zero
        assert jnp.all(buffer.info["obs"][1:] == 0)

    def test_increments_pos(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        for i in range(3):
            buffer = fns.add(buffer, _step(i, shapes), terminated=False)
        assert int(buffer.pos) == 3

    def test_wraps_position_modulo_size(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        # write size+2 = 10 transitions into a size-8 buffer
        for i in range(buffer.size + 2):
            buffer = fns.add(buffer, _step(i, shapes), terminated=False)
        # pos counter itself is monotonic (not wrapped)
        assert int(buffer.pos) == 10
        # but slots 0..1 now hold the *last two* transitions (i=8, i=9)
        assert float(buffer.info["obs"][0][0]) == 8.0
        assert float(buffer.info["obs"][1][0]) == 9.0
        # slot 7 still holds i=7 (most recent into upper half)
        assert float(buffer.info["obs"][7][0]) == 7.0

    def test_missing_keys_in_infos_leave_data_unchanged(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        # write a full transition
        buffer = fns.add(buffer, _step(1, shapes), terminated=False)
        # second write only updates "obs" — "act"/"rew" at slot 1 must remain 0
        buffer = fns.add(buffer, {"obs": jnp.full(shapes["obs"], 5.0)}, terminated=False)
        assert jnp.all(buffer.info["obs"][1] == 5.0)
        assert jnp.all(buffer.info["act"][1] == 0.0)
        assert float(buffer.info["rew"][1]) == 0.0


# ---------------------------------------------------------------------------
# sampling_ok: predecessor + terminal rules
# ---------------------------------------------------------------------------


class TestSamplingOk:
    def test_first_add_non_terminal_marks_nothing(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(0, shapes), terminated=False)
        # slot 0 has no known successor yet
        assert not bool(buffer.sampling_ok[0])
        # prev wraps to size-1 but had no predecessor to validate either
        assert not bool(buffer.sampling_ok[buffer.size - 1])
        # everything else also False
        assert not jnp.any(buffer.sampling_ok)

    def test_second_add_marks_previous_slot(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(0, shapes), terminated=False)
        buffer = fns.add(buffer, _step(1, shapes), terminated=False)
        # slot 0 now has a valid successor in slot 1
        assert bool(buffer.sampling_ok[0])
        # slot 1's "next" is still unknown
        assert not bool(buffer.sampling_ok[1])

    def test_terminal_slot_marked_immediately(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(0, shapes), terminated=True)
        assert bool(buffer.sampling_ok[0])

    def test_non_terminal_then_terminal(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = fns.add(buffer, _step(0, shapes), terminated=False)
        buffer = fns.add(buffer, _step(1, shapes), terminated=True)
        # slot 0 valid (successor exists), slot 1 valid (terminal)
        assert bool(buffer.sampling_ok[0])
        assert bool(buffer.sampling_ok[1])

    def test_predecessor_rule_survives_wraparound(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        # fill size+1 = 9 non-terminal transitions
        for i in range(buffer.size + 1):
            buffer = fns.add(buffer, _step(i, shapes), terminated=False)
        # pos==9; last write went to slot 0 (i=8). that should have marked slot 7
        # (its predecessor in ring) as valid. slot 0 itself is not yet valid:
        # its successor (slot 1, holding old i=1 data) was not just written.
        assert bool(buffer.sampling_ok[7])
        assert not bool(buffer.sampling_ok[0])


# ---------------------------------------------------------------------------
# sample: shapes, contents, validity
# ---------------------------------------------------------------------------


class TestSample:
    def _fill_with_terminals(self, fns, buffer, shapes, n):
        """Fill `n` transitions, all terminal so every slot is sampleable."""
        for i in range(n):
            buffer = fns.add(buffer, _step(i, shapes), terminated=True)
        return buffer

    def test_sample_returns_buffer_sample_and_indices(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = self._fill_with_terminals(fns, buffer, shapes, n=8)
        out, indices = fns.sample(buffer, jax.random.key(0))
        assert isinstance(out, BufferSample)
        # sampling_size=4 in the fixture
        assert indices.shape == (4,)

    def test_sample_this_info_keys_and_shapes(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = self._fill_with_terminals(fns, buffer, shapes, n=8)
        out, _ = fns.sample(buffer, jax.random.key(0))
        # this_step_infos = ["obs", "act", "rew"]
        assert set(out.this_info.keys()) == {"obs", "act", "rew"}
        assert out.this_info["obs"].shape == (4,) + shapes["obs"]
        assert out.this_info["act"].shape == (4,) + shapes["act"]
        assert out.this_info["rew"].shape == (4,)

    def test_sample_next_info_only_contains_next_keys(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = self._fill_with_terminals(fns, buffer, shapes, n=8)
        out, _ = fns.sample(buffer, jax.random.key(0))
        # next_step_infos = ["obs"]
        assert set(out.next_info.keys()) == {"obs"}
        assert out.next_info["obs"].shape == (4,) + shapes["obs"]

    def test_sample_next_obs_is_circular_successor(self, empty_buffer, shapes):
        buffer, fns = empty_buffer
        buffer = self._fill_with_terminals(fns, buffer, shapes, n=8)
        out, indices = fns.sample(buffer, jax.random.key(123))
        for batch_idx in range(indices.shape[0]):
            i = int(indices[batch_idx])
            j = (i + 1) % buffer.size
            assert jnp.allclose(out.this_info["obs"][batch_idx], buffer.info["obs"][i])
            assert jnp.allclose(out.next_info["obs"][batch_idx], buffer.info["obs"][j])

    def test_sample_only_draws_valid_slots(self, shapes):
        # size=4, only slot 2 valid → every drawn index must be 2.
        buffer, fns = create_buffer(
            shapes=shapes,
            size=4,
            sampling_size=16,
            this_step_infos=["obs"],
            next_step_infos=["obs"],
        )
        # write 3 non-terminal steps so slots 0,1 get marked valid (via predecessor
        # rule on next add). Then write step 3 as terminal so slot 2 valid too.
        # Mask everything off except slot 2 manually for a tighter test:
        for i in range(3):
            buffer = fns.add(buffer, _step(i, shapes), terminated=False)
        # mask: only slot 2 is valid
        forced = jnp.array([False, False, True, False])
        buffer = buffer._replace(sampling_ok=forced)
        _, indices = fns.sample(buffer, jax.random.key(7))
        assert jnp.all(indices == 2)


# ---------------------------------------------------------------------------
# create_sample: usable independently of create_buffer
# ---------------------------------------------------------------------------


class TestCreateSampleStandalone:
    def test_works_against_externally_built_buffer(self, shapes):
        size = 4
        # build a Buffer manually
        info = {k: jnp.arange(size * max(1, int(jnp.prod(jnp.asarray(shape)))))
                .reshape((size,) + shape)
                .astype(jnp.float32)
                for k, shape in shapes.items()}
        buffer = Buffer(
            info=info,
            sampling_ok=jnp.array([True, True, False, False]),
            pos=4,
            size=size,
        )
        sample_fn = create_sample(
            buffer_size=size,
            sampling_size=8,
            this_keys=["obs"],
            next_keys=["obs"],
        )
        out, indices = sample_fn(buffer, jax.random.key(0))
        assert out.this_info["obs"].shape == (8,) + shapes["obs"]
        # only slots 0,1 are valid; every drawn index must be 0 or 1
        assert jnp.all((indices == 0) | (indices == 1))


# ---------------------------------------------------------------------------
# create_sequence_sample: contiguous K-step window sampling
# ---------------------------------------------------------------------------


class TestCreateSequenceSample:
    def test_sequence_sample_shapes(self, shapes):
        buffer, fns = create_buffer(
            shapes=shapes,
            size=8,
            sampling_size=2,
            this_step_infos=["obs"],
            next_step_infos=["obs"],
        )
        for i in range(8):
            buffer = fns.add(buffer, _step(i, shapes), terminated=True)
        sample_fn = create_sequence_sample(
            buffer_size=8,
            sampling_size=2,
            sequence_size=3,
            keys=["obs"],
        )
        out, seq_indices = sample_fn(buffer, jax.random.key(0))
        # sampling_size=2, sequence_size=3
        assert seq_indices.shape == (2, 3)
        assert out.this_info["obs"].shape == (2, 3) + shapes["obs"]
        # content: row b column t holds the obs at slot seq_indices[b, t]
        for b in range(2):
            for t in range(3):
                slot = int(seq_indices[b, t])
                assert jnp.allclose(
                    out.this_info["obs"][b, t], buffer.info["obs"][slot]
                )

    def test_sequence_indices_are_contiguous(self, shapes):
        buffer, fns = create_buffer(
            shapes=shapes,
            size=8,
            sampling_size=4,
            this_step_infos=["obs"],
            next_step_infos=["obs"],
        )
        for i in range(8):
            buffer = fns.add(buffer, _step(i, shapes), terminated=True)
        sample_fn = create_sequence_sample(
            buffer_size=8,
            sampling_size=4,
            sequence_size=3,
            keys=["obs"],
        )
        _, seq_indices = sample_fn(buffer, jax.random.key(42))
        # each row should be [k, k+1, k+2] for some start k
        diffs = jnp.diff(seq_indices, axis=1)
        assert jnp.all(diffs == 1)

    def test_sequence_sample_respects_window_validity(self, shapes):
        # only an unbroken run of length sequence_size+1 starting at some slot
        # may be drawn. force a single valid run starting at slot 2.
        buffer, fns = create_buffer(
            shapes=shapes,
            size=8,
            sampling_size=16,
            this_step_infos=["obs"],
            next_step_infos=["obs"],
        )
        for i in range(8):
            buffer = fns.add(buffer, _step(i, shapes), terminated=True)
        # mask: only slots 2,3,4,5 valid — one length-4 contiguous run
        forced = jnp.array(
            [False, False, True, True, True, True, False, False]
        )
        buffer = buffer._replace(sampling_ok=forced)
        sample_fn = create_sequence_sample(
            buffer_size=8,
            sampling_size=16,
            sequence_size=3,
            keys=["obs"],
        )
        _, seq_indices = sample_fn(buffer, jax.random.key(0))
        # with sequence_size=3 the reduce-window length is sequence_size+1=4,
        # so only start=2 produces an all-valid window. every draw must start at 2.
        starts = seq_indices[:, 0]
        assert jnp.all(starts == 2)
