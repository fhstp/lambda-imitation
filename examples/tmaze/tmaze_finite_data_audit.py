"""Audit finite-data Retrace fits against an independent empirical fixed point.

    python examples/tmaze/tmaze_finite_data_audit.py
    python examples/tmaze/tmaze_finite_data_audit.py --training-checks

The empirical operator is assembled with forward eligibility traces in NumPy,
independently of the production reverse-scan target routine. It separates
finite-dataset error from neural optimization error without fitting to the
population oracle. Original results are read only.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tmaze_finite_data_detection as finite


def empirical_fixed_point(data, count, policy, gamma, trace):
    """Solve sum_{episodes,t} z_t delta_t(q) = 0 on visited coordinates.

    z_t = gamma * c_t * z_{t-1} + gamma**t * one_hot(o_t,a_t).
    This includes the same discounted initial-time distribution as the runner.
    Values with no empirical identification are not filled or regularized.
    """
    n_obs, n_actions = policy.shape
    obs, actions = (np.asarray(array[:, :count]) for array in data[:2])
    rewards, dones, ratios = (np.asarray(array[:, :count], dtype=np.float64)
                              for array in data[2:])
    steps, batch = obs.shape
    coordinates = obs * n_actions + actions
    active = np.unique(coordinates)
    lookup = np.full(n_obs * n_actions, -1, dtype=int)
    lookup[active] = np.arange(len(active))
    ids = lookup[coordinates]
    next_policy_features = np.zeros((n_obs, len(active)))
    active_obs, active_actions = np.divmod(active, n_actions)
    next_policy_features[active_obs, np.arange(len(active))] = policy[active_obs, active_actions]
    continuation_obs = obs[1:][dones[:-1] == 0]
    if not np.allclose(next_policy_features.sum(axis=1)[continuation_obs], 1):
        raise ValueError("Dataset does not cover a target bootstrap action; empirical root is unidentified")

    matrix = np.zeros((len(active), len(active)))
    rhs = np.zeros(len(active))
    counts = np.zeros(len(active))
    eligibility = np.zeros((batch, len(active)))
    identity = np.eye(len(active))
    for t in range(steps):
        if t:
            eligibility *= (1 - dones[t - 1])[:, None]
        eligibility *= (gamma * trace * np.minimum(1, ratios[t]))[:, None]
        eligibility[np.arange(batch), ids[t]] += gamma**t
        residual_features = identity[ids[t]].copy()
        if t + 1 < steps:
            residual_features -= gamma * (1 - dones[t])[:, None] * next_policy_features[obs[t + 1]]
        matrix += eligibility.T @ residual_features
        rhs += eligibility.T @ rewards[t]
        np.add.at(counts, ids[t], gamma**t)
    solution = np.linalg.solve(matrix, rhs)
    normalized_matrix = matrix / counts[:, None]
    normalized_rhs = rhs / counts
    q = np.full((n_obs, n_actions), np.nan)
    q[active_obs, active_actions] = solution
    residual = normalized_rhs - normalized_matrix @ solution
    return {
        "q": q, "active": active, "matrix": normalized_matrix, "rhs": normalized_rhs,
        "counts": counts,
        "condition_number": float(np.linalg.cond(normalized_matrix)),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(np.eye(len(active)) - normalized_matrix)))),
        "root_residual": float(np.max(np.abs(residual))),
    }


def error_summary(predictions, reference, weights):
    error = np.asarray(predictions) - np.asarray(reference)
    bias = error @ weights
    centered = error - bias[:, None]
    mse = np.mean(np.sum(weights * error**2, axis=-1))
    bias_mse = np.mean(bias**2)
    return {
        "rmse": float(np.sqrt(mse)),
        "signed_bias_by_lambda": bias.tolist(),
        "constant_component_rms": float(np.sqrt(bias_mse)),
        "centered_rms": float(np.sqrt(np.mean(np.sum(weights * centered**2, axis=-1)))),
        "constant_fraction_of_mse": float(bias_mse / mse) if mse > 1e-24 else 0.0,
    }


def verify_production_moments(data, count, reference, systems, config):
    """Check the independent roots using the actual reverse-scan targets."""
    obs, actions, rewards, dones, ratios = (
        np.asarray(array[:, :count]) for array in data
    )
    time_mass = config.gamma ** np.arange(obs.shape[0])[:, None]
    worst = 0.0
    for trace, system in zip(config.lambdas, systems):
        q = np.nan_to_num(system["q"]) * config.reward_scale
        taken = q[obs, actions]
        v = np.sum(q * reference["target"], axis=-1)[obs]
        targets = finite.retrace_targets(
            jnp.asarray(taken), jnp.asarray(v), jnp.asarray(rewards * config.reward_scale),
            jnp.asarray(dones), jnp.asarray(ratios), config.gamma, trace,
        )
        moments = np.zeros_like(q)
        masses = np.zeros_like(q)
        np.add.at(moments, (obs.ravel(), actions.ravel()),
                  (time_mass * (np.asarray(targets) - taken)).ravel())
        np.add.at(masses, (obs.ravel(), actions.ravel()), np.broadcast_to(time_mass, obs.shape).ravel())
        worst = max(worst, float(np.max(np.abs(moments[masses > 0] / masses[masses > 0]))))
    return worst


def audit_saved_run(result):
    values = result["protocol"]["config"].copy()
    values["lambdas"], values["hidden_dims"] = tuple(values["lambdas"]), tuple(values["hidden_dims"])
    config = finite.Config(**values)
    references = {eta: finite.reference_values(config, eta) for eta in result["protocol"]["etas"]}
    saved_references = {item["eta"]: item for item in result["references"]}
    datasets = {(seed, eta): finite.sample_episodes(config, eta, seed,
                                                   max(result["protocol"]["episode_budgets"]))
                for seed in result["protocol"]["seeds"] for eta in result["protocol"]["etas"]}
    audited = []
    for run in result["runs"]:
        seed, eta, count = run["seed"], run["eta"], run["episodes"]
        reference = references[eta]
        data = datasets[seed, eta]
        systems = [empirical_fixed_point(data, count, reference["target"], config.gamma, trace)
                   for trace in config.lambdas]
        coordinates = np.asarray(saved_references[eta]["scored_coordinates"])
        weights = np.asarray(saved_references[eta]["scored_weights"])
        empirical = np.asarray([system["q"][coordinates[:, 0], coordinates[:, 1]] for system in systems])
        truth = np.asarray(saved_references[eta]["scored_q_exact"])
        fitted = np.asarray(run["scored_q_estimates"])
        empirical_d = float(np.sqrt(np.sum(weights * (empirical[1] - empirical[0])**2)))
        record = {
            "seed": seed, "eta": eta, "episodes": count,
            "original_discrepancy": run["estimated_discrepancy"],
            "empirical_discrepancy": empirical_d,
            "population_discrepancy": reference["discrepancy"],
            "total_error": error_summary(fitted, truth, weights),
            "finite_dataset_error": error_summary(empirical, truth, weights),
            "optimization_error": error_summary(fitted, empirical, weights),
            "twin_gap_rms": run["twin_gap_rms"],
            "production_moment_residual_scaled": verify_production_moments(data, count, reference, systems, config),
            "empirical_q": empirical.tolist(),
            "max_system_condition_number": max(system["condition_number"] for system in systems),
            "max_operator_spectral_radius": max(system["spectral_radius"] for system in systems),
        }
        audited.append(record)
    return config, references, datasets, audited


def training_checks(config, result, references, datasets, audited, seeds):
    if config.settle_updates:
        raise ValueError("Continuation comparisons start from the original constant-rate run")
    count = max(result["protocol"]["episode_budgets"])
    checks = []
    first = references[result["protocol"]["etas"][0]]
    initialize, original_train, predict = finite.create_trainer(config, first["n_observations"], first["target"])
    continuation_rates = (config.learning_rate, config.learning_rate / 10, config.learning_rate / 100)
    trainers = {rate: finite.create_trainer(config._replace(learning_rate=rate),
                                            first["n_observations"], first["target"])[1]
                for rate in continuation_rates}
    for seed in seeds:
        for eta in result["protocol"]["etas"]:
            data = datasets[seed, eta]
            device_data = jax.tree.map(jnp.asarray, data)
            reference = references[eta]
            selected = reference["weights"] > 0
            weights = reference["weights"][selected]
            truth = reference["q"][:, selected]
            original = next(item for item in result["runs"]
                            if (item["seed"], item["eta"], item["episodes"]) == (seed, eta, count))
            empirical = np.asarray(next(item for item in audited
                                        if (item["seed"], item["eta"], item["episodes"]) == (seed, eta, count))["empirical_q"])
            state, _ = original_train(initialize(seed), device_data, jnp.asarray(count), steps=config.updates)
            fitted = np.asarray(predict(state.params)).min(axis=1)[:, selected] / config.reward_scale
            np.testing.assert_allclose(fitted, original["scored_q_estimates"], atol=2e-5, rtol=2e-5)
            print(f"Reproduced seed={seed}, eta={eta:g}, baseline empirical RMSE "
                  f"{error_summary(fitted, empirical, weights)['rmse']:.6f}", flush=True)
            for rate in continuation_rates:
                continued = state
                history = []
                for step in range(0, 8000, 500):
                    continued, _ = trainers[rate](continued, device_data, jnp.asarray(count), steps=500)
                    online = np.asarray(predict(continued.params)).min(axis=1)[:, selected] / config.reward_scale
                    target = np.asarray(predict(continued.targets)).min(axis=1)[:, selected] / config.reward_scale
                    history.append({"updates": config.updates + step + 500,
                                    "online": error_summary(online, empirical, weights),
                                    "target": error_summary(target, empirical, weights),
                                    "population_error": error_summary(online, truth, weights)})
                record = {"seed": seed, "eta": eta, "episodes": count,
                          "continuation_learning_rate": rate, "history": history}
                checks.append(record)
                final = history[-1]
                print(f"  +8000 steps, lr={rate:g}: empirical RMSE "
                      f"online={final['online']['rmse']:.6f}, target={final['target']['rmse']:.6f}; "
                      f"population={final['population_error']['rmse']:.6f}", flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tmaze_ld_output/finite_data/tmaze_finite_data_results.json"))
    parser.add_argument("--output", type=Path, default=Path("tmaze_ld_output/finite_data_audit.json"))
    parser.add_argument("--training-checks", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 4])
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    config, references, datasets, audited = audit_saved_run(result)
    for eta in result["protocol"]["etas"]:
        for count in result["protocol"]["episode_budgets"]:
            rows = [item for item in audited if item["eta"] == eta and item["episodes"] == count]
            means = [np.mean([item[key]["rmse"] for item in rows]) for key in
                     ("total_error", "finite_dataset_error", "optimization_error")]
            biases = np.mean([item["optimization_error"]["signed_bias_by_lambda"] for item in rows], axis=0)
            constant_fraction = np.mean([item["optimization_error"]["constant_component_rms"]**2 for item in rows])
            denominator = np.mean([item["optimization_error"]["rmse"]**2 for item in rows])
            constant_fraction /= max(denominator, 1e-30)
            print(f"eta={eta:g} episodes={count:4d}: RMSE total/data/opt "
                  f"{means[0]:.5f}/{means[1]:.5f}/{means[2]:.5f}; "
                  f"opt biases={biases}; constant MSE fraction={constant_fraction:.1%}")
    print("Max independent-root production residual:",
          max(item["production_moment_residual_scaled"] for item in audited))
    output = {"source": str(args.input), "config": config._asdict(), "audited_runs": audited}
    if args.training_checks:
        output["training_checks"] = training_checks(config, result, references, datasets, audited, args.seeds)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
