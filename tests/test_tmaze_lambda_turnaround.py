"""Regression checks for the on-policy λ-discrepancy turnaround construction."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


_examples = Path(__file__).resolve().parents[1] / "examples/tmaze"
with patch.object(sys, "path", [str(_examples), *sys.path]):
    import tmaze_lambda_turnaround as turnaround


def test_center_policy_is_exactly_zero_at_both_observation_endpoints():
    result = turnaround.run_sweep(points=101)
    center = result["dimensions"][0]["variants"][2]
    loss = np.asarray(center["loss"])
    assert loss[0] < 1e-24
    assert loss[-1] < 1e-24
    assert max(center["discrepancy"]) > 0.09
    assert center["peak_eta"] == pytest.approx(0.17, abs=0.02)


def test_nearby_policies_turn_around_without_hitting_zero_at_full_aliasing():
    result = turnaround.run_sweep(points=201)
    for dimension in result["dimensions"]:
        for variant in dimension["variants"]:
            if variant["label"] == "exact":
                continue
            assert variant["endpoint_discrepancy"] > 0.005
            assert variant["peak_discrepancy"] > variant["endpoint_discrepancy"] * 1.5
            assert 0.1 < variant["peak_eta"] < 0.25
            # There is a finite interval in which gradient descent in eta would
            # increase aliasing because the squared discrepancy is decreasing.
            assert variant["negative_derivative_fraction"] > 0.75


def test_reported_derivative_matches_a_fresh_central_difference():
    result = turnaround.run_sweep(points=401)
    eta = np.asarray(result["eta"])
    exact = result["dimensions"][0]["variants"][2]
    loss = np.asarray(exact["loss"])
    derivative = np.asarray(exact["loss_derivative"])
    central = (loss[2:] - loss[:-2]) / (eta[2:] - eta[:-2])
    np.testing.assert_allclose(derivative[1:-1], central, atol=2e-5, rtol=3e-3)
    assert np.any(derivative[1:-1] > 0)
    assert np.any(derivative[1:-1] < 0)


def test_center_policy_is_observation_law_invariant_under_the_channel():
    pomdp = turnaround.tm.TMazePOMDP(1)
    policy = turnaround.turnaround_policy()
    state_policy = pomdp.phi @ policy
    for eta in (0.0, 0.17, 0.5, 1.0):
        channel = turnaround.alias.observation_channel(pomdp, "both", eta)
        np.testing.assert_allclose(channel @ state_policy, state_policy, atol=1e-15)
