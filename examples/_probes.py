"""Optional post-hoc memory decoding, shared by Battleship and Minesweeper.

Probe labels are privileged diagnostic data. They never enter agent training,
action selection or replay. Train and test rollouts have independent RNG streams.
"""

from pathlib import Path
from typing import NamedTuple


class ProbeSpec(NamedTuple):
    truth: object
    observed: object
    shape: tuple
    title: str
    classes: int = 0  # 0 means binary cells; positive means categorical clues


def auroc(scores, labels):
    """Mann–Whitney AUROC with average ranks for ties."""
    import numpy as np

    scores, labels = np.asarray(scores).ravel(), np.asarray(labels, bool).ravel()
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    order = scores.argsort(kind="stable")
    _, starts, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(scores.size)
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def score_probe(targets, probs, observed, ages, spec):
    import numpy as np

    targets, probs = np.asarray(targets), np.asarray(probs)
    if spec.classes == 0:
        truth, pred = targets > 0.5, probs >= 0.5
        nll = -(targets * np.log2(np.clip(probs, 1e-7, 1))
                + (1 - targets) * np.log2(np.clip(1 - probs, 1e-7, 1)))
    else:
        targets = targets.reshape(*observed.shape, spec.classes)
        probs = probs.reshape(*observed.shape, spec.classes)
        truth, pred = targets.argmax(-1), probs.argmax(-1)
        nll = -(targets * np.log2(np.clip(probs, 1e-7, 1))).sum(-1)
    groups = {"all": np.ones_like(observed, bool),
              "observed": observed.astype(bool), "unobserved": ~observed.astype(bool)}
    result = {}
    for group, mask in groups.items():
        if not mask.any():
            continue
        valid_rows = mask.any(-1)
        error = (truth != pred) & mask
        metrics = {"accuracy": float((truth == pred)[mask].mean()),
                   "errors_per_state": float(error.sum(-1)[valid_rows].mean()),
                   "exact_match": float((~error.any(-1))[valid_rows].mean()),
                   "bits_per_cell": float(nll[mask].mean())}
        recalls = []
        for c in range(max(2, spec.classes)):
            selected = mask & (truth == c)
            recall = float((pred[selected] == c).mean()) if selected.any() else float("nan")
            metrics[f"recall_{c}"] = recall
            recalls.append(recall)
        metrics["balanced"] = float(np.mean(recalls))
        if spec.classes == 0:
            metrics["auroc"] = auroc(probs[mask], truth[mask])
        result.update({f"{group}/{k}": v if np.isfinite(v) else None for k, v in metrics.items()})
    for low, high in ((0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 1000000)):
        mask = (ages >= low) & (ages < high)
        if mask.any():
            result[f"retention/age_{low}_{high}"] = float((pred[mask] == truth[mask]).mean())
    return result


def render_probe(path, data, probabilities, spec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def grid(x):
        return x.reshape(spec.shape)

    ends = np.flatnonzero(data["dones"])
    length = int(ends[0]) + 1 if ends.size else len(data["targets"])
    frames = np.unique(np.linspace(0, length - 1, min(6, length), dtype=int))
    fig, axes = plt.subplots(2, len(frames), squeeze=False, figsize=(2.5 * len(frames), 5))
    for col, frame in enumerate(frames):
        truth, prediction = data["targets"][frame], probabilities[frame]
        vmax = max(1, spec.classes - 1)
        if spec.classes > 0:
            truth = truth.reshape(-1, spec.classes).argmax(-1)
            prediction = prediction.reshape(-1, spec.classes) @ np.arange(spec.classes)
        for row, values in enumerate((truth, prediction)):
            ax = axes[row, col]
            ax.imshow(grid(values), vmin=0, vmax=vmax, cmap="viridis")
            # Outline observed cells; never overwrite predictions with labels.
            seen = np.argwhere(grid(data["observed"][frame]).astype(float) == 1)
            if len(seen):
                ax.scatter(seen[:, 1], seen[:, 0], marker="s", facecolors="none",
                           edgecolors="white", s=35, linewidths=0.6)
            ax.set_title(f"{'Truth' if row == 0 else spec.title}, t={frame}", fontsize=9)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_probes(args, experiment, fns, agents, carry_dim, rnd):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    env, spec, params = experiment.env, experiment.probe, experiment.env.default_params
    action_dim = env.action_space(params).n
    _, dummy = env.reset(jax.random.key(0), params)
    n_outputs, n_cells = spec.truth(dummy).size, spec.observed(dummy).size

    @jax.jit
    def collect(agent, key):
        key, rk = jax.random.split(key)
        obs, es = env.reset(rk, params)

        def step(carry, _):
            obs, es, memory, pa, ages, key = carry
            key, ak, ek = jax.random.split(key, 3)
            targets, observed = spec.truth(es), spec.observed(es)
            ages = jnp.where(observed, ages + 1, -1)
            action, memory = fns.predict(agent, obs, memory, ak, prev_action=pa)
            obs, es, _, done, _ = env.step(ek, es, action.astype(jnp.int32), params)
            sample = {"carries": memory, "targets": targets, "observed": observed,
                      "ages": ages, "dones": done}
            pa = fns.encode_action(action)
            return (obs, es, jnp.where(done, 0, memory), jnp.where(done, 0, pa),
                    jnp.where(done, -1, ages), key), sample

        initial = (obs, es, jnp.zeros(carry_dim), jnp.zeros(action_dim),
                   -jnp.ones(n_cells, jnp.int32), key)
        return jax.lax.scan(step, initial, length=args.probe_collect_steps)[1]

    def forward(p, x):
        for w, b in p[:-1]:
            x = jax.nn.relu(x @ w + b)
        return x @ p[-1][0] + p[-1][1]

    @jax.jit
    def fit(data, key):
        init_keys = jax.random.split(key, 4)
        dims = (carry_dim, args.probe_hidden_dim, args.probe_hidden_dim, n_outputs)
        p = tuple((jax.random.normal(k, (a, b)) * (2 / a) ** 0.5, jnp.zeros(b))
                  for k, a, b in zip(init_keys, dims[:-1], dims[1:]))
        opt = optax.adam(args.probe_lr)

        def step(carry, _):
            p, state, key = carry
            key, bk = jax.random.split(key)
            idx = jax.random.randint(bk, (args.probe_batch_size,), 0, data["carries"].shape[0])

            def loss(p):
                logits, target = forward(p, data["carries"][idx]), data["targets"][idx]
                if spec.classes == 0:
                    return optax.sigmoid_binary_cross_entropy(logits, target).mean()
                return optax.softmax_cross_entropy(logits.reshape(-1, spec.classes),
                                                    target.reshape(-1, spec.classes)).mean()

            grads = jax.grad(loss)(p)
            updates, state = opt.update(grads, state)
            return (optax.apply_updates(p, updates), state, key), None

        return jax.lax.scan(step, (p, opt.init(p), init_keys[-1]), length=args.probe_steps)[0][0]

    results = []
    output = Path(args.output_dir) / f"probe_{rnd}"
    output.mkdir(exist_ok=True)
    for i in range(args.num_seeds):
        key = jax.random.fold_in(jax.random.fold_in(jax.random.key(args.seed + i), 100), rnd)
        train_key, test_key, fit_key = jax.random.split(key, 3)
        agent = jax.tree.map(lambda x: x[i], agents)
        training, test = collect(agent, train_key), collect(agent, test_key)
        p = fit(training, fit_key)
        logits = forward(p, test["carries"])
        probabilities = (jax.nn.sigmoid(logits) if spec.classes == 0 else
                         jax.nn.softmax(logits.reshape(-1, n_cells, spec.classes), -1).reshape(logits.shape))
        data, probabilities = jax.device_get((test, probabilities))
        metrics = score_probe(data["targets"], probabilities, data["observed"], data["ages"], spec)
        results.append(metrics)
        np.savez_compressed(output / f"seed_{i}.npz", **data, probabilities=probabilities)
        render_probe(output / f"seed_{i}.png", data, probabilities, spec)
    combined = {}
    for name in set().union(*(m.keys() for m in results)):
        values = [m.get(name) for m in results]
        combined.update({f"probe/{name}/seed_{i}": value for i, value in enumerate(values)})
        valid = [v for v in values if v is not None]
        combined[f"probe/{name}/mean"] = float(np.mean(valid)) if valid else None
    return combined
