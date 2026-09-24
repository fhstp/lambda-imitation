"""Every parameter a W&B sweep varies must reach Hyperparameters.

A sweep that names a parameter the script does not expose is silently ignored:
the agent reports "ignoring unknown parameter", every trial runs the default,
and the sweep looks perfectly healthy while measuring nothing.  That happened
with ``online_batch_size``, which was pinned to ``--batch-size`` and has since
been merged into it: there is now exactly one batch knob, ``batch_size``, the
number of sequences drawn from the online buffer per gradient step.

This drives the *real* CLI and the *real* ``hp = Hyperparameters(...)`` block
of the probe scripts (lifted from their source), so it fails if either drifts.
"""

import shlex
import sys
from pathlib import Path

import pytest

from lambda_imitation.iqlearn import Hyperparameters

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "lambda-envs"

# (sweep parameter, Hyperparameters field, probe value)
SWEPT = [
    ("alpha", "alpha", "0.37"),
    ("batch_size", "batch_size", "123"),
    ("burn_in_length", "burn_in_length", "7"),
    ("lambda1", "lambda1", "0.23"),
    ("lambda2", "lambda2", "0.66"),
    ("lambda_coef", "lambda_coef", "0.077"),
    ("online_buffer_size", "online_buffer_size", "150000"),
    ("sequence_length", "sequence_length", "44"),
    ("gamma", "gamma", "0.91"),
    ("tau", "tau", "0.004"),
]

def _build(script, extra):
    """Run the script's own CLI + hp block; return (args, hyperparameters)."""
    src = (EXAMPLES / script).read_text()
    cli = src[:src.index("# ── always-needed imports")]
    start = src.index("    hp = Hyperparameters(")
    end = src.index("\n    )", start) + len("\n    )")
    hp_block = "\n".join(line[4:] for line in src[start:end].split("\n"))

    saved = sys.argv
    try:
        sys.argv = ["probe", "--vis-only"] + shlex.split(extra)
        g = {"__name__": "sweep_check", "__file__": str(EXAMPLES / script)}
        exec(compile(cli, "cli", "exec"), g)
        ns = {"args": g["args"], "Hyperparameters": Hyperparameters}
        exec(compile(hp_block, "hp", "exec"), ns)
        return g["args"], ns["hp"]
    finally:
        sys.argv = saved

@pytest.mark.parametrize("name,field,value", SWEPT)


def test_pocman_sweep_parameter_reaches_hyperparameters(name, field, value):
    args, hp = _build("pocman_pellet_probe.py", f"--{name}={value}")
    # W&B emits the underscore spelling; parse_args rewrites it to the flag
    assert getattr(args, name) == type(getattr(args, name))(value)
    assert getattr(hp, field) == type(getattr(hp, field))(value)


def test_there_is_exactly_one_batch_knob():
    """``online_batch_size`` is gone and ``batch_size`` is the live one.

    It used to be the other way round: ``batch_size`` sized an expert buffer
    that nothing sampled, so setting it changed nothing at all, while the real
    knob had no flag of its own.
    """
    from lambda_imitation.iqlearn import Hyperparameters

    assert "batch_size" in Hyperparameters._fields
    assert "online_batch_size" not in Hyperparameters._fields
    _, hp = _build("pocman_pellet_probe.py", "--batch-size=77")
    assert hp.batch_size == 77


def test_battleship_exposes_the_same_knobs():
    _, hp = _build("battleship_board_probe.py", "--batch-size=77 --lambda-coef=0.5")
    assert (hp.batch_size, hp.lambda_coef) == (77, 0.5)
