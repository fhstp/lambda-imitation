"""Tests for the circular replay buffer.

The first class (TestSamplingOk) is a direct conversion of the old __main__
inline tests to the new generic-info-dict API.  The remaining classes cover
behaviour that was previously untested.
"""

import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.buffer import Buffer, SequenceSample, create_buffer, create_sequence_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPES = {"obs": (3,), "act": (2,), "rew": ()}


def make_buffer(size=5, sampling_size=1):
    """Create a buffer with the same logical schema the old tests used."""
    return create_buffer(
        shapes=SHAPES,
        size=size,
        sampling_size=sampling_size,
        this_step_infos=["obs", "act", "rew"],
        next_step_infos=["obs"],
    )


def add_step(buffer, fns, obs, act, rew, terminated):
    """Thin wrapper matching the old positional add() signature."""
    return fns.add(
        buffer,
        {"obs": jnp.array(obs, dtype=jnp.float32),
         "act": jnp.array(act, dtype=jnp.float32),
         "rew": jnp.array(rew, dtype=jnp.float32)},
        terminated,
    )


# ---------------------------------------------------------------------------
# Converted old __main__ tests  (sampling_ok flag + stored data)
# ---------------------------------------------------------------------------

class TestSamplingOk:
    """sampling_ok flag logic -- direct port of the seven original tests."""

    def test_single_non_terminal(self):
        buf, fns = make_buffer()
        buf = add_step(buf, fns, [1, 1, 1], [1, 1], 1.0, False)

        assert not buf.sampling_ok.any()
        assert (buf.info["obs"][0] == jnp.array([1.0, 1.0, 1.0])).all()
        assert (buf.info["obs"][1:] == 0).all()
        assert (buf.info["act"][0] == jnp.array([1.0, 1.0])).all()
        assert (buf.info["act"][1:] == 0).all()
        assert (buf.info["rew"] == jnp.array([1.0, 0, 0, 0, 0])).all()

    def test_single_terminal(self):
        buf, fns = make_buffer()
        buf = add_step(buf, fns, [1, 1, 1], [1, 1], 1.0, True)

        expected = jnp.array([True, False, False, False, False])
        assert (buf.sampling_ok == expected).all()

    def test_two_non_terminal(self):
        buf, fns = make_buffer()
        buf = add_step(buf, fns, [1, 1, 1], [1, 1], 1.0, False)
        buf = add_step(buf, fns, [2, 2, 2], [2, 2], 2.0, False)

        assert (buf.sampling_ok == jnp.array([True, False, False, False, False])).all()
        assert (buf.info["obs"][0] == jnp.array([1.0, 1.0, 1.0])).all()
        assert (buf.info["obs"][1] == jnp.array([2.0, 2.0, 2.0])).all()
        assert (buf.info["obs"][2:] == 0).all()
        assert (buf.info["act"][0] == jnp.array([1.0, 1.0])).all()
        assert (buf.info["act"][1] == jnp.array([2.0, 2.0])).all()
        assert (buf.info["act"][2:] == 0).all()
        assert (buf.info["rew"] == jnp.array([1.0, 2.0, 0, 0, 0])).all()

    def test_full_buffer(self):
        buf, fns = make_buffer()
        for i in range(5):
            v = float(i + 1)
            buf = add_step(buf, fns, [v, v, v], [v, v], v, False)

        assert (buf.sampling_ok == jnp.array([True, True, True, True, False])).all()
        for i in range(5):
            v = float(i + 1)
            assert (buf.info["obs"][i] == jnp.full((3,), v)).all()
            assert (buf.info["act"][i] == jnp.full((2,), v)).all()
        assert (buf.info["rew"] == jnp.arange(1, 6, dtype=jnp.float32)).all()

    def test_circular_wrap(self):
        """6 adds into size-5 buffer; write head wraps to slot 0."""
        buf, fns = make_buffer()
        for i in range(6):
            v = float(i + 1)
            buf = add_step(buf, fns, [v, v, v], [v, v], v, False)

        assert (buf.sampling_ok == jnp.array([False, True, True, True, True])).all()
        # slot 0 overwritten by step 6
        assert (buf.info["obs"][0] == jnp.full((3,), 6.0)).all()
        assert (buf.info["act"][0] == jnp.full((2,), 6.0)).all()
        assert float(buf.info["rew"][0]) == 6.0
        # slots 1-4 still hold original data
        for i in range(1, 5):
            v = float(i + 1)
            assert (buf.info["obs"][i] == jnp.full((3,), v)).all()

    def test_circular_wrap_plus_one(self):
        """7 adds into size-5 buffer; write head at slot 1."""
        buf, fns = make_buffer()
        for i in range(7):
            v = float(i + 1)
            buf = add_step(buf, fns, [v, v, v], [v, v], v, False)

        assert (buf.sampling_ok == jnp.array([True, False, True, True, True])).all()
        assert (buf.info["obs"][0] == jnp.full((3,), 6.0)).all()
        assert (buf.info["obs"][1] == jnp.full((3,), 7.0)).all()
        for i in range(2, 5):
            assert (buf.info["obs"][i] == jnp.full((3,), float(i + 1))).all()

    def test_circular_terminal(self):
        """6 adds, last one terminal: all slots become sampleable."""
        buf, fns = make_buffer()
        for i in range(6):
            v = float(i + 1)
            buf = add_step(buf, fns, [v, v, v], [v, v], v, i == 5)

        assert (buf.sampling_ok == jnp.array([True, True, True, True, True])).all()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_initial_state(self):
        buf, _ = make_buffer(size=4, sampling_size=2)
        assert buf.pos == 0
        assert buf.size == 4
        assert not buf.sampling_ok.any()
        for k, shape in SHAPES.items():
            assert buf.info[k].shape == (4,) + shape
            assert (buf.info[k] == 0).all()

    def test_all_shape_keys_present(self):
        buf, _ = make_buffer()
        assert set(buf.info.keys()) == set(SHAPES.keys())


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

class TestPosTracking:
    def test_pos_increments(self):
        buf, fns = make_buffer()
        assert buf.pos == 0
        for i in range(10):
            buf = add_step(buf, fns, [0, 0, 0], [0, 0], 0.0, False)
            assert buf.pos == i + 1

    def test_size_unchanged_after_wraps(self):
        buf, fns = make_buffer(size=3)
        for _ in range(7):
            buf = add_step(buf, fns, [0, 0, 0], [0, 0], 0.0, False)
        assert buf.size == 3


# ---------------------------------------------------------------------------
# Partial add (only some info keys supplied)
# ---------------------------------------------------------------------------

class TestPartialAdd:
    def test_missing_key_preserves_zeros(self):
        buf, fns = make_buffer()
        buf = fns.add(buf, {"obs": jnp.array([1.0, 2.0, 3.0])}, False)

        assert (buf.info["obs"][0] == jnp.array([1.0, 2.0, 3.0])).all()
        # act and rew were not supplied -- should stay zero
        assert (buf.info["act"][0] == jnp.zeros(2)).all()
        assert float(buf.info["rew"][0]) == 0.0

    def test_overwrite_single_key(self):
        """Add full step, then overwrite only one key at the same slot."""
        buf, fns = make_buffer()
        buf = add_step(buf, fns, [1, 1, 1], [1, 1], 1.0, False)
        buf = add_step(buf, fns, [2, 2, 2], [2, 2], 2.0, False)
        # Now overwrite slot 2 (pos=2) with only obs
        buf = fns.add(buf, {"obs": jnp.array([9.0, 9.0, 9.0])}, False)
        assert (buf.info["obs"][2] == jnp.full((3,), 9.0)).all()
        # act and rew at slot 2 remain zero (never written)
        assert (buf.info["act"][2] == jnp.zeros(2)).all()


# ---------------------------------------------------------------------------
# Episode boundaries
# ---------------------------------------------------------------------------

class TestEpisodeBoundary:
    def test_terminal_then_new_episode(self):
        """Terminal step followed by the start of a new episode."""
        buf, fns = make_buffer()
        # Episode 1: step 0 (non-terminal), step 1 (terminal)
        buf = add_step(buf, fns, [1, 1, 1], [1, 1], 1.0, False)
        buf = add_step(buf, fns, [2, 2, 2], [2, 2], 2.0, True)
        assert (buf.sampling_ok == jnp.array([True, True, False, False, False])).all()

        # Episode 2: step 2 (non-terminal)
        buf = add_step(buf, fns, [3, 3, 3], [3, 3], 3.0, False)
        # slot 1 still sampleable (was terminal; now also has valid next)
        # slot 2 non-terminal with no next yet
        assert (buf.sampling_ok == jnp.array([True, True, False, False, False])).all()

    def test_two_complete_episodes(self):
        """Two 2-step episodes back to back."""
        buf, fns = make_buffer()
        buf = add_step(buf, fns, [1, 0, 0], [0, 0], 0.0, False)
        buf = add_step(buf, fns, [2, 0, 0], [0, 0], 1.0, True)
        buf = add_step(buf, fns, [3, 0, 0], [0, 0], 0.0, False)
        buf = add_step(buf, fns, [4, 0, 0], [0, 0], 1.0, True)

        # 0: has next -> T, 1: terminal -> T, 2: has next -> T, 3: terminal -> T
        assert (buf.sampling_ok == jnp.array([True, True, True, True, False])).all()


# ---------------------------------------------------------------------------
# Small buffer edge cases
# ---------------------------------------------------------------------------

class TestSmallBuffer:
    def test_size_two_fill(self):
        buf, fns = create_buffer(
            shapes={"x": (1,)}, size=2, sampling_size=1,
            this_step_infos=["x"], next_step_infos=["x"],
        )
        buf = fns.add(buf, {"x": jnp.array([1.0])}, False)
        assert not buf.sampling_ok.any()

        buf = fns.add(buf, {"x": jnp.array([2.0])}, False)
        # slot 0 has valid next (slot 1); slot 1 does not
        assert (buf.sampling_ok == jnp.array([True, False])).all()

    def test_size_two_wrap(self):
        buf, fns = create_buffer(
            shapes={"x": (1,)}, size=2, sampling_size=1,
            this_step_infos=["x"], next_step_infos=["x"],
        )
        for i in range(3):
            buf = fns.add(buf, {"x": jnp.array([float(i)])}, False)
        # 3 adds into size 2: slot 0 overwritten, write head back at slot 1
        assert (buf.sampling_ok == jnp.array([False, True])).all()

    def test_size_three_multiple_wraps(self):
        buf, fns = create_buffer(
            shapes={"x": ()}, size=3, sampling_size=1,
            this_step_infos=["x"], next_step_infos=["x"],
        )
        for i in range(8):
            buf = fns.add(buf, {"x": jnp.array(float(i))}, False)
        # pos=8; last written slot is (8-1)%3 = 1 (pos is post-increment)
        # slot 1 is freshest (no valid next yet) -> F
        # slot 2: has next at 0 -> T, slot 0: has next at 1 -> T
        assert (buf.sampling_ok == jnp.array([True, False, True])).all()


# ---------------------------------------------------------------------------
# Heterogeneous shapes
# ---------------------------------------------------------------------------

class TestHeterogeneousShapes:
    def test_mixed_shapes(self):
        """Buffer with scalar, 1-D, and 2-D info entries."""
        buf, fns = create_buffer(
            shapes={"scalar": (), "vec": (4,), "mat": (2, 3)},
            size=3, sampling_size=1,
            this_step_infos=["scalar", "vec", "mat"],
            next_step_infos=["vec"],
        )
        assert buf.info["scalar"].shape == (3,)
        assert buf.info["vec"].shape == (3, 4)
        assert buf.info["mat"].shape == (3, 2, 3)

        buf = fns.add(
            buf,
            {"scalar": jnp.array(5.0),
             "vec": jnp.ones(4),
             "mat": jnp.ones((2, 3)) * 2},
            False,
        )
        assert float(buf.info["scalar"][0]) == 5.0
        assert (buf.info["vec"][0] == 1.0).all()
        assert (buf.info["mat"][0] == 2.0).all()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    @pytest.fixture()
    def filled_buffer(self):
        """Size-5 buffer fully filled (non-terminal), sampling_size=3."""
        buf, fns = make_buffer(size=5, sampling_size=3)
        for i in range(5):
            v = float(i + 1)
            buf = add_step(buf, fns, [v, v, v], [v, v], v, False)
        return buf, fns

    def test_output_shapes(self, filled_buffer):
        buf, fns = filled_buffer
        key = jax.random.key(42)
        sample, indices = fns.sample(buf, key)

        assert sample.this_info["obs"].shape == (3, 3)   # (sampling_size, obs_dim)
        assert sample.this_info["act"].shape == (3, 2)
        assert sample.this_info["rew"].shape == (3,)
        assert sample.next_info["obs"].shape == (3, 3)
        assert indices.shape == (3,)

    def test_this_info_keys(self, filled_buffer):
        buf, fns = filled_buffer
        sample, _ = fns.sample(buf, jax.random.key(0))
        assert set(sample.this_info.keys()) == {"obs", "act", "rew"}

    def test_next_info_keys(self, filled_buffer):
        buf, fns = filled_buffer
        sample, _ = fns.sample(buf, jax.random.key(0))
        assert set(sample.next_info.keys()) == {"obs"}

    def test_sampled_indices_are_valid(self, filled_buffer):
        """Every sampled index must have sampling_ok = True."""
        buf, fns = filled_buffer
        for seed in range(5):
            _, indices = fns.sample(buf, jax.random.key(seed))
            for idx in indices:
                assert buf.sampling_ok[idx], f"seed={seed}, idx={int(idx)}"

    def test_this_next_pairing(self, filled_buffer):
        """sample.next_info[i] should hold data from slot (index+1) % size."""
        buf, fns = filled_buffer
        sample, indices = fns.sample(buf, jax.random.key(7))
        for i, idx in enumerate(indices):
            next_idx = (int(idx) + 1) % buf.size
            expected = buf.info["obs"][next_idx]
            assert (sample.next_info["obs"][i] == expected).all()

    def test_this_data_matches_buffer(self, filled_buffer):
        """sample.this_info[i] should hold data from slot index."""
        buf, fns = filled_buffer
        sample, indices = fns.sample(buf, jax.random.key(99))
        for i, idx in enumerate(indices):
            assert (sample.this_info["obs"][i] == buf.info["obs"][idx]).all()
            assert (sample.this_info["act"][i] == buf.info["act"][idx]).all()
            assert sample.this_info["rew"][i] == buf.info["rew"][idx]

    def test_different_keys_give_different_samples(self, filled_buffer):
        """Sanity: two different PRNG keys should (almost certainly) give different indices."""
        buf, fns = filled_buffer
        _, idx_a = fns.sample(buf, jax.random.key(0))
        _, idx_b = fns.sample(buf, jax.random.key(1))
        # With 4 valid slots and sampling_size=3 the chance of identical draws is tiny
        assert not (idx_a == idx_b).all()


# ---------------------------------------------------------------------------
# Stage C: create_sequence_sample + SequenceSample
# ---------------------------------------------------------------------------

_SEQ_SIZE  = 10   # buffer capacity
_SEQ_T     = 3    # sequence length
_SEQ_BATCH = 4    # sequences per call
_OBS_DIM   = 2    # observation shape


def _make_seq_buffer(size=_SEQ_SIZE, terminal_slots=None):
    """Fill a buffer storing 'obs' and 'terminal' keys.

    All slots are filled in order.  terminal_slots is a set of indices at
    which terminated=True is passed (default: only the last slot).
    """
    if terminal_slots is None:
        terminal_slots = {size - 1}
    buf, fns = create_buffer(
        shapes={"obs": (_OBS_DIM,), "terminal": ()},
        size=size,
        sampling_size=1,
        this_step_infos=["obs", "terminal"],
        next_step_infos=["obs"],
    )
    for i in range(size):
        terminated = i in terminal_slots
        buf = fns.add(
            buf,
            {
                "obs": jnp.array([float(i), float(i)]),
                "terminal": jnp.array(float(terminated)),
            },
            terminated,
        )
    return buf


def _seq_sampler(size=_SEQ_SIZE, T=_SEQ_T, batch=_SEQ_BATCH):
    return create_sequence_sample(
        buffer_size=size,
        batch_size=batch,
        sequence_length=T,
        keys=["obs", "terminal"],
        terminal_key="terminal",
    )


class TestSequenceSampleShapes:
    """create_sequence_sample returns correct shapes and types."""

    def test_info_shape(self):
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(0))
        assert sample.info["obs"].shape == (_SEQ_BATCH, _SEQ_T, _OBS_DIM)
        assert sample.info["terminal"].shape == (_SEQ_BATCH, _SEQ_T)

    def test_mask_shape(self):
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(0))
        assert sample.mask.shape == (_SEQ_BATCH, _SEQ_T)

    def test_start_index_shape(self):
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(0))
        assert sample.start_index.shape == (_SEQ_BATCH,)

    def test_mask_dtype(self):
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(0))
        assert sample.mask.dtype == jnp.float32

    def test_result_is_sequence_sample(self):
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(0))
        assert isinstance(sample, SequenceSample)


class TestSequenceSampleValidity:
    """All sampled start positions must be valid (sampling_ok) for the full window."""

    def test_all_starts_sampling_ok(self):
        buf = _make_seq_buffer()
        sample_fn = _seq_sampler()
        for seed in range(10):
            sample = sample_fn(buf, jax.random.key(seed))
            for s in sample.start_index:
                assert buf.sampling_ok[int(s)], f"seed={seed}, start={int(s)}"

    def test_all_window_slots_sampling_ok(self):
        """Every slot in each sampled T-length window must have sampling_ok=True."""
        buf = _make_seq_buffer()
        sample_fn = _seq_sampler()
        for seed in range(5):
            sample = sample_fn(buf, jax.random.key(seed))
            for s in sample.start_index:
                for t in range(_SEQ_T):
                    slot = (int(s) + t) % _SEQ_SIZE
                    assert buf.sampling_ok[slot], (
                        f"seed={seed}, start={int(s)}, offset={t}, slot={slot}"
                    )

    def test_unfilled_tail_never_sampled(self):
        """Starts that would extend into unwritten slots must never be chosen."""
        # Fill only 5 of 10 slots (all non-terminal), then sample T=3.
        # Slot 4 has no valid next, so only starts {0, 1} are valid.
        buf, fns = create_buffer(
            shapes={"obs": (_OBS_DIM,), "terminal": ()},
            size=10,
            sampling_size=1,
            this_step_infos=["obs", "terminal"],
            next_step_infos=["obs"],
        )
        for i in range(5):
            buf = fns.add(
                buf,
                {"obs": jnp.array([float(i), float(i)]), "terminal": jnp.array(0.0)},
                False,
            )
        # sampling_ok: [T, T, T, T, F, F, F, F, F, F]
        sample_fn = create_sequence_sample(
            buffer_size=10,
            batch_size=16,
            sequence_length=3,
            keys=["obs", "terminal"],
            terminal_key="terminal",
        )
        for seed in range(10):
            sample = sample_fn(buf, jax.random.key(seed))
            for s in sample.start_index:
                assert int(s) in (0, 1), f"unexpected start={int(s)}"


class TestSequenceSampleMask:
    """Terminal mask correctness."""

    def test_no_terminal_mask_all_ones(self):
        """Sequence with no terminal inside: mask must be all 1s."""
        # terminal_slots={9}: terminal only at slot 9; T=3 sequences starting at 0..6
        # never include slot 9 if start <= 6 (0+3-1=2, ..., 6+3-1=8).
        buf = _make_seq_buffer(terminal_slots={9})
        sample_fn = create_sequence_sample(
            buffer_size=_SEQ_SIZE,
            batch_size=20,
            sequence_length=3,
            keys=["obs", "terminal"],
            terminal_key="terminal",
        )
        # Force starts that don't include slot 9: all valid starts are 0..7
        # (0+2=2, 7+2=9 — slot 9 IS included when start=7).
        # Use start=0 explicitly by checking we get mask=1 at least sometimes.
        sample = sample_fn(buf, jax.random.key(0))
        # Sequences that don't touch the terminal (not starting at 7) must be all-1 masked.
        for b in range(sample.mask.shape[0]):
            s = int(sample.start_index[b])
            window = [(s + t) % _SEQ_SIZE for t in range(3)]
            if 9 not in window:
                assert jnp.all(sample.mask[b] == 1.0), (
                    f"start={s}, window={window}, mask={sample.mask[b]}"
                )

    def test_terminal_at_start_mask(self):
        """Terminal at t=0: mask[0]=1, mask[1:]=0."""
        # Make a buffer where slot 0 is terminal, all others non-terminal (except last).
        # Sample T=3 starting at slot 0.
        size = 6
        buf, fns = create_buffer(
            shapes={"obs": (_OBS_DIM,), "terminal": ()},
            size=size,
            sampling_size=1,
            this_step_infos=["obs", "terminal"],
            next_step_infos=["obs"],
        )
        for i in range(size):
            terminated = (i == 0) or (i == size - 1)
            buf = fns.add(
                buf,
                {"obs": jnp.array([float(i), float(i)]),
                 "terminal": jnp.array(float(i == 0))},
                terminated,
            )
        sample_fn = create_sequence_sample(
            buffer_size=size,
            batch_size=20,
            sequence_length=3,
            keys=["obs", "terminal"],
            terminal_key="terminal",
        )
        sample = sample_fn(buf, jax.random.key(42))
        for b in range(sample.mask.shape[0]):
            if int(sample.start_index[b]) == 0:
                expected = jnp.array([1.0, 0.0, 0.0])
                assert jnp.allclose(sample.mask[b], expected), (
                    f"mask={sample.mask[b]}"
                )

    def test_terminal_mid_sequence_mask(self):
        """Terminal at t=1: mask=[1,1,0]."""
        # Slot 2 is terminal; sequences starting at slot 1 = [slot1, slot2(term), slot3]
        size = 8
        T = 3
        buf, fns = create_buffer(
            shapes={"obs": (_OBS_DIM,), "terminal": ()},
            size=size,
            sampling_size=1,
            this_step_infos=["obs", "terminal"],
            next_step_infos=["obs"],
        )
        for i in range(size):
            terminated = (i == 2) or (i == size - 1)
            buf = fns.add(
                buf,
                {"obs": jnp.array([float(i), float(i)]),
                 "terminal": jnp.array(float(i == 2))},
                terminated,
            )
        sample_fn = create_sequence_sample(
            buffer_size=size,
            batch_size=20,
            sequence_length=T,
            keys=["obs", "terminal"],
            terminal_key="terminal",
        )
        sample = sample_fn(buf, jax.random.key(7))
        for b in range(sample.mask.shape[0]):
            if int(sample.start_index[b]) == 1:
                # window = [slot1, slot2(term), slot3] → mask = [1, 1, 0]
                expected = jnp.array([1.0, 1.0, 0.0])
                assert jnp.allclose(sample.mask[b], expected), (
                    f"mask={sample.mask[b]}"
                )

    def test_data_matches_buffer_at_gathered_positions(self):
        """info['obs'][b, t] must equal buffer.info['obs'][(start+t)%size]."""
        buf = _make_seq_buffer()
        sample = _seq_sampler()(buf, jax.random.key(3))
        for b in range(_SEQ_BATCH):
            for t in range(_SEQ_T):
                slot = (int(sample.start_index[b]) + t) % _SEQ_SIZE
                expected = buf.info["obs"][slot]
                assert jnp.allclose(sample.info["obs"][b, t], expected), (
                    f"b={b}, t={t}, slot={slot}"
                )
