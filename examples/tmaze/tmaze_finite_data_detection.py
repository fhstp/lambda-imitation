"""Estimate Retrace discrepancy from finite T-maze replay with frozen policies.

    python examples/tmaze/tmaze_finite_data_detection.py
    python examples/tmaze/tmaze_finite_data_detection.py --seeds 0 1 \
        --episodes 64 512 --updates 1000 --output-dir /tmp/tmaze-pilot
    python examples/tmaze/tmaze_finite_data_detection.py \
        --settle-updates 8000 --settle-learning-rate 1e-5 \
        --output-dir tmaze_ld_output/finite_data_settled

Uses the production Head, Retrace target routine, twin-min bootstraps, detached
Huber regression, Adam, and EMA targets. The representation is a fixed one-hot
observation. Data budgets restart from the same initialization and receive the
same update budget. Complete episodes avoid trace-tail truncation. Discounted
loss weights match the exact evaluator's starting-state occupancy posterior.

Rewards are scaled for fitting; all reported values use original reward units.
The Huber linear-region fraction is recorded: an active robust loss need not
have the expected-Retrace fixed point as its population optimum. Exact values
are used only for evaluation, never for training or stopping.
"""

import argparse
from fractions import Fraction
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from lambda_imitation.actor_critic import Head, retrace_targets
import tmaze_lambda_discrepancy as tm
import tmaze_retrace_aliasing as alias


class Config(NamedTuple):
    hallway_length: int = 5
    gamma: float = 0.9
    lambdas: tuple[float, float] = (0.0, 1.0)
    behaviour_up: float = 0.5
    target_up: float = 2 / 3
    aliasing: str = "both"
    hidden_dims: tuple[int, ...] = (64, 64)
    batch_size: int = 128
    learning_rate: float = 1e-3
    tau: float = 0.005
    reward_scale: float = 0.1
    huber_delta: float = 1.0
    loss: str = "huber"
    updates: int = 4000
    eval_every: int = 500
    settle_updates: int = 0
    settle_learning_rate: float = 1e-5


class Episodes(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    ratios: np.ndarray


class CriticState(NamedTuple):
    params: object
    targets: object
    opt_state: object
    key: jax.Array


def sample_episodes(config, eta, seed, count):
    """Time-major complete episodes; budgets are prefixes of the same dataset.

    Independent PRNG streams pair physical paths and emission uniforms across
    representation conditions without exposing latent states to the learner.
    """
    model = tm.TMazePOMDP(config.hallway_length, gamma=config.gamma)
    steps = model.L + 2
    goals = np.random.default_rng(np.random.SeedSequence([seed, 0])).integers(0, 2, count)
    junction_draws = np.random.default_rng(np.random.SeedSequence([seed, 1])).random(count)
    junction_actions = np.where(junction_draws < config.behaviour_up, tm.A_NORTH, tm.A_SOUTH)
    physical_states = goals[:, None] * steps + np.arange(steps)[None, :]
    channel = alias.observation_channel(model, config.aliasing, eta)
    uniforms = np.random.default_rng(np.random.SeedSequence([seed, 2])).random((count, steps))
    cumulative = np.cumsum(channel[physical_states], axis=-1)
    observations = np.sum(uniforms[..., None] >= cumulative, axis=-1).astype(np.int32)
    assert np.all(observations < model.n_states)
    actions = np.full((count, steps), tm.A_EAST, dtype=np.int32)
    actions[:, -1] = junction_actions
    rewards = np.zeros((count, steps), dtype=np.float32)
    rewards[:, -1] = np.where(junction_actions == goals, model.good_reward, model.bad_reward)
    dones = np.zeros_like(rewards)
    dones[:, -1] = 1
    ratios = np.ones_like(rewards)
    ratios[:, -1] = np.where(
        junction_actions == tm.A_NORTH,
        config.target_up / config.behaviour_up,
        (1 - config.target_up) / (1 - config.behaviour_up),
    )
    return Episodes(*(np.ascontiguousarray(array.T) for array in
                      (observations, actions, rewards, dones, ratios)))


def reference_values(config, eta):
    model = tm.TMazePOMDP(config.hallway_length, gamma=config.gamma)
    channel = alias.observation_channel(model, config.aliasing, eta)
    target = model.phi @ tm.junction_policy(config.target_up)
    behavior = model.phi @ tm.junction_policy(config.behaviour_up)
    np.testing.assert_allclose(channel @ target, target, atol=1e-14)
    np.testing.assert_allclose(channel @ behavior, behavior, atol=1e-14)
    augmented = alias.observation_augmented_model(model, channel)
    evaluator = tm.RetraceEvaluator(augmented, behavior)
    q_values = np.asarray(tm.value_pair(evaluator, target, config.lambdas))
    weights = channel.T @ alias.reference_pair_weights(model)
    score = alias.weighted_gap(q_values[0], q_values[1], weights)
    return {"q": q_values, "weights": weights, "target": target,
            "discrepancy": float(score), "n_observations": model.n_states}


def create_trainer(config, n_observations, target_policy):
    """Functional critic-only version of the production Retrace regression."""
    if config.loss != "huber":
        raise ValueError("the publication protocol uses Huber regression")
    prototype = Head(n_observations, config.hidden_dims, tm.NA, rngs=nnx.Rngs(0))
    graph, _ = nnx.split(prototype, nnx.Param)
    features = jnp.eye(n_observations, dtype=jnp.float32)
    policy = jnp.asarray(target_policy, dtype=jnp.float32)
    lambdas = jnp.asarray(config.lambdas, dtype=jnp.float32)
    learning_rate = (optax.piecewise_constant_schedule(
        config.learning_rate, {config.updates: config.settle_learning_rate / config.learning_rate}
    ) if config.settle_updates else config.learning_rate)
    optimizer = optax.adam(learning_rate)
    # Uniform episodes, with time weighting to match discounted-occupancy W_mu.
    time_weights = config.gamma ** jnp.arange(config.hallway_length + 2)
    time_weights /= time_weights.mean()

    def initialize(seed):
        initial_key = jax.random.fold_in(jax.random.key(seed), 3)
        head_states = []
        for key in jax.random.split(initial_key, 4):
            head = Head(n_observations, config.hidden_dims, tm.NA, rngs=nnx.Rngs(key))
            _, state = nnx.split(head, nnx.Param)
            head_states.append(state)
        params = jax.tree.map(lambda *arrays: jnp.stack(arrays), *head_states)
        return CriticState(params, params, optimizer.init(params),
                           jax.random.fold_in(jax.random.key(seed), 4))

    def all_values(params):
        # The one-hot representation has only n_observations distinct inputs.
        # Evaluating them once is identical to evaluating every repeated input.
        values = jax.vmap(lambda state: nnx.merge(graph, state)(features))(params)
        return values.reshape(2, 2, n_observations, tm.NA)

    def loss_fn(params, targets, batch):
        target_values = jnp.min(all_values(targets), axis=1)
        target_taken = target_values[:, batch.observations, batch.actions]
        target_v = jnp.sum(target_values * policy[None, :, :], axis=-1)[:, batch.observations]
        returns = jax.vmap(lambda q, v, lam: retrace_targets(
            q, v, batch.rewards * config.reward_scale, batch.dones, batch.ratios,
            config.gamma, lam,
        ))(target_taken, target_v, lambdas)
        returns = jax.lax.stop_gradient(returns)
        predictions = all_values(params)[:, :, batch.observations, batch.actions]
        errors = predictions - returns[:, None]
        losses = optax.huber_loss(predictions, returns[:, None], delta=config.huber_delta)
        # Average both twins' regression losses separately for each lambda, then sum.
        loss = (losses * time_weights[None, None, :, None]).mean(axis=(1, 2, 3)).sum()
        stats = {"loss": loss,
                 "max_abs_regression_error_scaled": jnp.max(jnp.abs(errors)),
                 "huber_linear_fraction": jnp.mean(jnp.abs(errors) > config.huber_delta)}
        return loss, stats

    @partial(jax.jit, static_argnames=("steps",))
    def train_chunk(state, data, n_episodes, *, steps):
        def update(carry, _):
            key, sample_key = jax.random.split(carry.key)
            indices = jax.random.randint(sample_key, (config.batch_size,), 0, n_episodes)
            batch = jax.tree.map(lambda value: value[:, indices], data)
            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                carry.params, carry.targets, batch)
            updates, opt_state = optimizer.update(grads, carry.opt_state, carry.params)
            params = optax.apply_updates(carry.params, updates)
            targets = optax.incremental_update(params, carry.targets, config.tau)
            return CriticState(params, targets, opt_state, key), metrics

        state, history = jax.lax.scan(update, state, None, length=steps)
        stats = {"loss": history["loss"].mean(),
                 "max_abs_regression_error_scaled": history["max_abs_regression_error_scaled"].max(),
                 "huber_linear_fraction": history["huber_linear_fraction"].mean()}
        return state, stats

    return initialize, train_chunk, jax.jit(all_values)


def evaluate_predictions(twin_values_scaled, reference, data, count, reward_scale):
    twins = np.asarray(twin_values_scaled, dtype=np.float64) / reward_scale
    q = twins.min(axis=1)
    weights = reference["weights"]
    selected = weights > 0
    selected_weights = weights[selected]
    truth = reference["q"][:, selected]
    predictions = q[:, selected]
    assert np.all(np.isfinite(truth)) and np.all(np.isfinite(predictions))
    estimate = float(np.sqrt(np.sum(selected_weights * (predictions[1] - predictions[0])**2)))
    errors = predictions - truth
    q_rmse = np.sqrt(np.sum(selected_weights[None, :] * errors**2, axis=1))
    biases = errors @ selected_weights
    centered_rmse = np.sqrt(np.sum(selected_weights * (errors - biases[:, None])**2, axis=1))
    discrepancy_errors = errors[1] - errors[0]
    visits = np.zeros_like(weights, dtype=np.int64)
    np.add.at(visits, (data.observations[:, :count].ravel(), data.actions[:, :count].ravel()), 1)
    twin_gap = twins[:, 0, selected] - twins[:, 1, selected]
    return {
        "estimated_discrepancy": estimate,
        "exact_discrepancy": reference["discrepancy"],
        "discrepancy_absolute_error": abs(estimate - reference["discrepancy"]),
        "q_rmse_by_lambda": q_rmse.tolist(),
        "q_bias_by_lambda": biases.tolist(),
        "q_centered_rmse_by_lambda": centered_rmse.tolist(),
        "q_rmse": float(np.sqrt(np.mean(q_rmse**2))),
        "discrepancy_vector_rmse": float(np.sqrt(np.sum(selected_weights * discrepancy_errors**2))),
        "discrepancy_vector_bias": float(discrepancy_errors @ selected_weights),
        "twin_gap_rms": float(np.sqrt(np.mean(np.sum(selected_weights * twin_gap**2, axis=-1)))),
        "scored_weight_coverage": float(weights[visits > 0].sum()),
        "minimum_scored_visits": int(visits[selected].min()),
        "scored_q_estimates": predictions.tolist(),
    }


def serializable_reference(reference):
    observations, actions = np.nonzero(reference["weights"] > 0)
    return {"exact_discrepancy": reference["discrepancy"],
            "scored_coordinates": np.column_stack((observations, actions)).tolist(),
            "scored_weights": reference["weights"][observations, actions].tolist(),
            "scored_q_exact": reference["q"][:, observations, actions].tolist()}


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_results(result, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_experiment(config, seeds, etas, budgets, output_dir, *, resume=False, run_label="experiment"):
    path = output_dir / "tmaze_finite_data_results.json"
    metadata = json.loads(json.dumps({"config": config._asdict(), "seeds": seeds,
                                      "etas": etas, "episode_budgets": budgets}))
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; choose another directory or use --resume")
        result = json.loads(path.read_text())
        if result["protocol"] != metadata:
            raise ValueError("Resume requires identical configuration, seeds, etas, and budgets")
        current_hashes = {
            "production_module_sha256": source_hash(inspect.getfile(retrace_targets)),
            "runner_sha256": source_hash(__file__),
            "exact_model_sha256": source_hash(tm.__file__),
            "observation_channel_sha256": source_hash(alias.__file__),
        }
        if any(result["software"].get(key) != value for key, value in current_hashes.items()):
            raise ValueError("Source files changed since this run; use a new output directory")
    else:
        result = {
            "schema_version": 1, "run_label": run_label, "protocol": metadata,
            "sampling": "complete forward episodes; gamma^t-weighted initial-state regression",
            "budget_protocol": "nested datasets; fresh matched initialization and fixed updates per budget",
            "score": "fixed reference latent-pair RMS weights transported through the observation channel",
            "reported_units": "original rewards (good=4, bad=-0.1)",
            "software": {"jax": jax.__version__, "devices": [str(device) for device in jax.devices()],
                         "production_module_sha256": source_hash(inspect.getfile(retrace_targets)),
                         "runner_sha256": source_hash(__file__),
                         "exact_model_sha256": source_hash(tm.__file__),
                         "observation_channel_sha256": source_hash(alias.__file__)},
            "references": [], "runs": [],
        }

    references = {eta: reference_values(config, eta) for eta in etas}
    result["references"] = [{"eta": eta, **serializable_reference(references[eta])} for eta in etas]
    first_reference = references[etas[0]]
    initialize, train_chunk, predict = create_trainer(
        config, first_reference["n_observations"], first_reference["target"])
    completed = {(item["seed"], item["eta"], item["episodes"]) for item in result["runs"]}
    for seed in seeds:
        initial = initialize(seed)
        for eta in etas:
            data = sample_episodes(config, eta, seed, max(budgets))
            device_data = jax.tree.map(jnp.asarray, data)
            reference = references[eta]
            for count in budgets:
                if (seed, eta, count) in completed:
                    continue
                start = time.perf_counter()
                state = initial
                history = []
                max_error, linear_mass, loss_mass = 0.0, 0.0, 0.0
                total_updates = config.updates + config.settle_updates
                for done in range(0, total_updates, config.eval_every):
                    chunk = min(config.eval_every, total_updates - done)
                    state, stats = train_chunk(state, device_data, jnp.asarray(count), steps=chunk)
                    stats = jax.tree.map(lambda value: float(np.asarray(value)), stats)
                    if not all(np.isfinite(value) for value in stats.values()):
                        raise FloatingPointError("Non-finite training metrics")
                    metrics = evaluate_predictions(predict(state.params), reference, data, count,
                                                   config.reward_scale)
                    target_metrics = evaluate_predictions(predict(state.targets), reference, data, count,
                                                          config.reward_scale)
                    metrics.update({"target_q_rmse": target_metrics["q_rmse"],
                                    "target_estimated_discrepancy": target_metrics["estimated_discrepancy"]})
                    history.append({"updates": done + chunk, **stats,
                                    "learning_rate_at_last_update": (
                                        config.learning_rate if done + chunk <= config.updates
                                        else config.settle_learning_rate),
                                    **{key: value for key, value in metrics.items()
                                       if key != "scored_q_estimates"}})
                    max_error = max(max_error, stats["max_abs_regression_error_scaled"])
                    linear_mass += chunk * stats["huber_linear_fraction"]
                    loss_mass += chunk * stats["loss"]
                record = {"seed": seed, "eta": eta, "episodes": count,
                          "transitions": count * (config.hallway_length + 2),
                          "updates": total_updates, "elapsed_seconds": time.perf_counter() - start,
                          "mean_training_loss": loss_mass / total_updates,
                          "huber_linear_fraction": linear_mass / total_updates,
                          "max_abs_regression_error_scaled": max_error,
                          **metrics, "history": history}
                result["runs"].append(record)
                save_results(result, path)
                print(f"seed={seed} eta={eta:g} N={record['transitions']:6d} "
                      f"D={metrics['estimated_discrepancy']:.4f} "
                      f"exact={reference['discrepancy']:.4f} "
                      f"Q-RMSE={metrics['q_rmse']:.4f} "
                      f"Huber-linear={record['huber_linear_fraction']:.3%} "
                      f"({record['elapsed_seconds']:.1f}s)", flush=True)
    return result


def make_figure(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    etas = result["protocol"]["etas"]
    seeds = result["protocol"]["seeds"]
    budgets = result["protocol"]["episode_budgets"]
    steps = result["protocol"]["config"]["hallway_length"] + 2
    colors = ("#2876a5", "#c96521", "#238b67", "#8c6bb1", "#a6761d")
    with plt.rc_context({"font.family": "serif", "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
        handles = []
        for i, eta in enumerate(etas):
            color = colors[i % len(colors)]
            rows = sorted((item for item in result["runs"] if item["eta"] == eta),
                          key=lambda item: (item["episodes"], item["seed"]))
            if any(sum(item["episodes"] == n for item in rows) != len(seeds) for n in budgets):
                raise ValueError("A plot requires every requested seed at every budget")
            label = ("Fully observed" if eta == 0 else "Aliased" if eta == 1 else "Mixed")
            handles.append(Line2D([0], [0], color=color, lw=1.6, marker="o", markersize=3,
                                  label=rf"{label} ($\eta={eta:g}$)"))
            for ax, field in zip(axes, ("estimated_discrepancy", "q_rmse")):
                values = np.asarray([[item[field] for item in rows if item["episodes"] == n]
                                     for n in budgets])
                mean = values.mean(axis=1)
                sd = values.std(axis=1, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
                x = np.asarray(budgets) * steps
                ax.plot(x, mean, color=color, marker="o", markersize=3, lw=1.6)
                ax.fill_between(x, np.maximum(0, mean - sd), mean + sd, color=color, alpha=0.16, lw=0)
            exact = next(item["exact_discrepancy"] for item in result["references"] if item["eta"] == eta)
            axes[0].axhline(exact, color=color, ls="--", lw=0.9, alpha=0.8)
        for ax in axes:
            ax.set_xscale("log")
            transition_budgets = np.asarray(budgets) * steps
            ax.set_xticks(transition_budgets, labels=[f"{n:,}" for n in transition_budgets])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel("Behaviour transitions collected", fontsize=8.5, labelpad=3)
            ax.tick_params(labelsize=8, width=0.6, length=2.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#e6e9ec", lw=0.55)
            ax.set_axisbelow(True)
        axes[0].set_title("(a) Estimated discrepancy", loc="left", fontsize=10, fontweight="bold")
        axes[1].set_title("(b) Critic estimation error", loc="left", fontsize=10, fontweight="bold")
        axes[0].set_ylabel(r"$\widehat D_{\mathrm{RMS}}$", fontsize=10)
        axes[1].set_ylabel("Q-value RMSE", fontsize=9)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.005),
                   ncol=min(3, len(handles)), frameon=False, fontsize=8,
                   handlelength=1.4, columnspacing=1)
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.29, top=0.85, wspace=0.35)
        stem = output_dir / "tmaze_finite_data_detection"
        fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(stem.with_suffix(".png"), dpi=240)
        plt.close(fig)


def probability(text):
    value = float(Fraction(text))
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("probability must lie in [0,1]")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--etas", type=probability, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--episodes", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--hallway-length", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lambdas", type=probability, nargs=2, default=[0.0, 1.0])
    parser.add_argument("--behaviour-up", type=probability, default=0.5)
    parser.add_argument("--target-up", type=probability, default=2 / 3)
    parser.add_argument("--aliasing", choices=alias.ALIAS_TYPES, default="both")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--reward-scale", type=float, default=0.1)
    parser.set_defaults(loss="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--settle-updates", type=int, default=0,
                        help="Additional low-step-size updates after the initial fit")
    parser.add_argument("--settle-learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("tmaze_ld_output/finite_data"))
    parser.add_argument("--run-label", default="experiment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if (not 0 < args.gamma < 1 or not 0 < args.behaviour_up < 1
            or not 0 < args.tau <= 1 or args.lambdas[0] == args.lambdas[1]):
        parser.error("require gamma in (0,1), covered behaviour in (0,1), tau in (0,1], and distinct lambdas")
    if (min(args.episodes) <= 0 or args.hallway_length < 1 or args.batch_size < 1
            or args.updates < 1 or args.eval_every < 1 or args.learning_rate <= 0
            or args.settle_updates < 0 or args.settle_learning_rate <= 0
            or args.reward_scale <= 0 or args.huber_delta <= 0
            or any(width < 1 for width in args.hidden_dims) or min(args.seeds) < 0):
        parser.error("budgets, dimensions, rates, and intervals must be positive; seeds must be nonnegative")
    if (len(set(args.seeds)) != len(args.seeds) or len(set(args.etas)) != len(args.etas)
            or sorted(set(args.episodes)) != args.episodes):
        parser.error("seeds and etas must be unique; episode budgets must be strictly increasing")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        result = json.loads((args.output_dir / "tmaze_finite_data_results.json").read_text())
    else:
        fields = {key: getattr(args, key) for key in Config._fields}
        fields["lambdas"] = tuple(fields["lambdas"])
        fields["hidden_dims"] = tuple(fields["hidden_dims"])
        result = run_experiment(Config(**fields), args.seeds, args.etas, args.episodes,
                                args.output_dir, resume=args.resume, run_label=args.run_label)
    if not args.no_figures:
        make_figure(result, args.output_dir)
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
