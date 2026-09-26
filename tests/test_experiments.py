"""Shared CLI, exact checkpoint continuation, and diagnostic isolation."""

import importlib
import json
from pathlib import Path
import pickle
import sys
from unittest.mock import patch

import jax
import numpy as np
import pytest
import yaml

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
    import _common as common
    import _probes as probes
    import battleship
    import minesweeper


@pytest.mark.parametrize("module", [battleship, minesweeper])
@pytest.mark.parametrize("flag", ["--retrace", "--gvd", "--offline", "--grad-clip",
                                  "--per-alpha", "--stop-actor-fe", "--use-sac"])
def test_removed_flags_are_rejected(module, flag):
    with pytest.raises(SystemExit):
        common.parse_args(module.parser(), [flag])


@pytest.mark.parametrize("method,auxiliary,coef", [("ac", False, 0), ("critics", True, 0), ("ld", True, 0.2)])
def test_method_configuration(method, auxiliary, coef):
    p = battleship.parser()
    args = common.parse_args(p, ["--method", method, "--lambda_coef=0.2"])
    common.validate_args(p, args)
    assert (args.method != "ac") == auxiliary
    assert args.lambda_coef == coef


def test_every_sweep_parameter_is_a_live_flag_and_spaces_match():
    configs = [yaml.safe_load(path.read_text()) for path in sorted((EXAMPLES / "sweeps").glob("*.yaml"))]
    assert len(configs) == 2
    actions = {a.dest: a for a in minesweeper.parser()._actions}
    for config in configs:
        assert config["program"] == "examples/minesweeper.py"
        assert set(config["parameters"]) <= actions.keys()
        p = minesweeper.parser()
        args = common.parse_args(p, [])
        for name, value in config["parameters"].items():
            setattr(args, name, value.get("value", value.get("min")))
        common.validate_args(p, args)
        assert args.prefill_steps == 3008
        assert args.rounds * args.train_steps == 200000
    shared = set(configs[0]["parameters"]) & set(configs[1]["parameters"])
    for name in shared - {"method", "lambda_coef"}:
        assert configs[0]["parameters"][name] == configs[1]["parameters"][name]


def small_args(module, output, rounds=1, extra=()):
    flags = ["--rounds", str(rounds), "--train-steps", "2", "--num-seeds", "2",
             "--memory-hidden-dim", "4", "--batch-size", "2", "--burn-in-length", "1",
             "--sequence-length", "2", "--lambda-truncation", "2", "--online-buffer-size", "64",
             "--prefill-steps", "16", "--eval-episodes", "2", "--checkpoint-every", "0",
             "--output-dir", str(output), *extra]
    if module is minesweeper:
        flags += ["--rows", "4", "--cols", "4", "--mines", "2", "--projection-dim", "4", "--head-dim", "4"]
    p = module.parser()
    return p, common.parse_args(p, flags)


def test_resume_reproduces_uninterrupted_training_exactly(tmp_path):
    p, full = small_args(minesweeper, tmp_path / "full", rounds=2)
    common.run(p, full, minesweeper.build_experiment)
    p, partial = small_args(minesweeper, tmp_path / "partial")
    common.run(p, partial, minesweeper.build_experiment)
    p, resumed = small_args(minesweeper, tmp_path / "resumed", rounds=2,
                            extra=("--resume-from", str(partial.output_dir / "checkpoint_final.pkl")))
    summary = common.run(p, resumed, minesweeper.build_experiment)
    assert summary["env_interactions"] == 20  # per-seed count includes random prefill
    with (full.output_dir / "checkpoint_final.pkl").open("rb") as f:
        expected = pickle.load(f)
    with (resumed.output_dir / "checkpoint_final.pkl").open("rb") as f:
        got = pickle.load(f)
    for a, b in zip(jax.tree.leaves(expected["training"]), jax.tree.leaves(got["training"])):
        np.testing.assert_array_equal(a, b)
    bad_config = {**got["config"], "alpha": 0.7}
    with pytest.raises(ValueError, match="same training configuration"):
        common.load_checkpoint(resumed.output_dir / "checkpoint_final.pkl", bad_config)


def test_probe_is_posthoc_and_produces_portable_artifacts(tmp_path):
    p, args = small_args(battleship, tmp_path / "probe", rounds=0,
                         extra=("--probe", "--probe-collect-steps", "8", "--probe-steps", "2",
                                "--probe-hidden-dim", "4", "--probe-batch-size", "2"))
    common.run(p, args, battleship.build_experiment)
    for i in range(2):
        assert (args.output_dir / "probe_0" / f"seed_{i}.png").exists()
        with np.load(args.output_dir / "probe_0" / f"seed_{i}.npz") as data:
            assert data["targets"].shape == (8, 25)
            assert np.isfinite(data["probabilities"]).all()
    history = [json.loads(row) for row in (args.output_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(any(k.startswith("probe/") for k in row) for row in history)
    with (args.output_dir / "checkpoint_final.pkl").open("rb") as f:
        saved = pickle.load(f)
    # Probe fitting must never apply agent updates.
    np.testing.assert_array_equal(saved["training"][0].update_step, [0, 0])


def test_probe_auroc_ties_and_class_imbalance():
    assert probes.auroc([0, 0, 0, 0], [0, 0, 0, 1]) == 0.5
    assert probes.auroc([0, 0, 0, 1], [0, 0, 0, 1]) == 1.0
    spec = probes.ProbeSpec(None, None, (2, 2), "test")
    targets = np.array([[0, 0, 0, 1]])
    result = probes.score_probe(targets, np.zeros_like(targets, float),
                                 np.ones_like(targets, bool), np.zeros_like(targets), spec)
    assert result["all/accuracy"] == 0.75
    assert result["all/balanced"] == 0.5
    assert result["all/errors_per_state"] == 1
