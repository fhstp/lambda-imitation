"""Read-only checkpoint diagnostics: fit only auxiliary critics on fixed replay.

No policy, memory, main critic, replay, or production source changes. Uses the
production Retrace loss through a guarded closure adapter (not a second loss
implementation). Held-out windows occupy a disjoint chronological replay region.
Residuals are NOT value RMSE against an oracle or proof of memory sufficiency.
"""

import argparse
from datetime import datetime
from functools import partial
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import NamedTuple

from popgym_alpha_sweep import file_hash
from popgym_suite import write_json


ENVS = ("popgym-HigherLowerHard-v0", "popgym-BattleshipEasy-v0")
MILESTONES = (0, 250, 1000, 4000, 8000, 12000)
SWEEP_UNIT = "lambda-popgym-alpha-scheduled.service"


class Heads(NamedTuple):
    q1: object
    q2: object
    target1: object
    target2: object
    opt1: object
    opt2: object
    step: object


def production_helpers(fns):
    """Fail loudly if closure structure changes; production files stay frozen."""
    def captures(fn):
        fn = inspect.unwrap(fn)
        result = {}
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            try:
                result[name] = cell.cell_contents
            except ValueError:
                pass  # Disabled optional GVD branches have empty closure cells.
        return result
    update = captures(fns.update)
    combined = captures(update["loss_combined"])
    names = ("loss_vtrace_lambda_sequence", "lambda1_critic_graph", "lambda2_critic_graph", "loss_ld", "params")
    result = {name: combined[name] for name in names}
    low = captures(result["loss_vtrace_lambda_sequence"])
    result["run_critic"] = captures(low["get_q_both"])["run_critic"]
    return result


def build_agent(config, port):
    import jax.numpy as jnp
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import EnvSpec, create_iqlearn_from_env
    hp = Hyperparameters(batch_size=1, online_batch_size=config["batch_size"], online_buffer_size=config["online_buffer_size"],
        fe_lr=config["fe_lr"], actor_lr=config["actor_lr"], critic_lr=config["critic_lr"], lambda_critic_lr=config["critic_lr"],
        alpha=config["alpha"], gamma=config["gamma"], tau=config["tau"], burn_in_length=config["burn_in_length"],
        sequence_length=config["sequence_length"], lambda_truncation=config["lambda_truncation"],
        lambda1=config["lambda1"], lambda2=config["lambda2"], lambda_coef=config["lambda_coef"],
        fake_onpolicy_loss=False, retrace=True, stop_actor_fe=True)
    return create_iqlearn_from_env(EnvSpec(port.obs_shape, port.action_dim, True),
        {"observations": jnp.zeros((1, *port.obs_shape)), "actions": jnp.zeros((1, 1))}, buffer_size=1, hp=hp,
        projection=config["projection_dim"] or None, memory_type=config["memory_type"],
        memory_hidden_dim=config["memory_hidden_dim"], use_prev_action=True,
        actor_dims=(config["head_dim"],), critic_dims=(config["head_dim"],),
        lambda1_critic_dims=(config["head_dim"],), lambda2_critic_dims=(config["head_dim"],),
        approximate_lambda=True, critic_layer_norm=config["critic_layer_norm"], debug=True,
        seed=config["seed"], use_sac=config["use_sac"])


def split_replay(buffer, window, train_count, test_count, seed):
    """Chronological ring indices; terminal split; no shared transition slots."""
    import numpy as np
    count, size, pos = min(int(buffer.pos), int(buffer.size)), int(buffer.size), int(buffer.pos)
    order = np.arange(pos - count, pos) % size
    done = np.asarray(buffer.info["terminated"])[order].astype(bool)
    candidates = np.flatnonzero(done[:int(count * .8)]) + 1
    candidates = candidates[(candidates > window + 1) & (count - candidates > window + 1)]
    if not len(candidates):
        raise ValueError("No natural episode boundary for disjoint replay regions")
    split = int(candidates[-1])
    valid = np.asarray(buffer.sampling_ok)[order].astype(bool)
    rng = np.random.default_rng(seed)
    selections, details = {}, {}
    for label, lo, hi, n in (("train", 0, split, train_count), ("holdout", split, count, test_count)):
        # The production sampler checks window+1 sampling_ok flags.
        sums = np.convolve(valid[lo:hi].astype(int), np.ones(window + 1, int), mode="valid")
        starts = np.flatnonzero(sums == window + 1) + lo
        if not len(starts):
            raise ValueError(f"No valid {label} sequences")
        if n > len(starts):
            raise ValueError(f"Requested more unique {label} windows than available")
        chosen = rng.choice(starts, n, replace=False)
        indices = order[chosen[:, None] + np.arange(window)]
        selections[label] = indices
        details[label] = {"chronological_interval": [lo, hi], "available_windows": len(starts),
                          "selected_windows": n, "unique_transition_slots": len(np.unique(indices)),
                          "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
    if np.intersect1d(selections["train"], selections["holdout"]).size:
        raise AssertionError("Training/holdout overlap")
    return selections, {"replay_transitions": count, "split_after_terminal": split, "window": window, **details}


def raw_windows(buffer, indices):
    import numpy as np
    keys = ("observations", "actions", "rewards", "terminated", "behaviour_weight")
    return {k: np.asarray(buffer.info[k])[indices] for k in keys}


def cache_features(state, debug, raw, config, chunk=64):
    """Fixed online/target feature caches, with production burn-in/reset rules."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    carry_dim = config["memory_hidden_dim"] * (2 if config["memory_type"] == "lstm" else 1)
    if config["memory_type"] == "identity":
        carry_dim = 0
    @jax.jit
    def encode(obs, actions, done):
        return debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
            obs, actions, done, jnp.zeros((obs.shape[0], carry_dim), jnp.float32))
    online, target = [], []
    for offset in range(0, len(raw["observations"]), chunk):
        a, b = jax.device_get(encode(*(jnp.asarray(raw[k][offset:offset + chunk])
                                    for k in ("observations", "actions", "terminated"))))
        online.append(np.swapaxes(a, 0, 1)); target.append(np.swapaxes(b, 0, 1))
    burn = config["burn_in_length"]
    return {"latent": jnp.asarray(np.concatenate(online)), "target_latent": jnp.asarray(np.concatenate(target)),
            **{k: jnp.asarray(raw[k][:, burn:]) for k in ("actions", "rewards", "terminated", "behaviour_weight")}}


def initial_heads(state):
    return Heads(state.lambda1_critic, state.lambda2_critic, state.lambda1_critic_target,
        state.lambda2_critic_target, state.lambda1_critic_optimizer_state, state.lambda2_critic_optimizer_state, state.update_step)


def with_heads(state, heads):
    # Nothing else, including update_step/replay, is changed in the full snapshot.
    return state._replace(lambda1_critic=heads.q1, lambda2_critic=heads.q2,
        lambda1_critic_target=heads.target1, lambda2_critic_target=heads.target2,
        lambda1_critic_optimizer_state=heads.opt1, lambda2_critic_optimizer_state=heads.opt2)


def make_refitter(state, helpers, dataset, batch_size):
    import jax
    import jax.numpy as jnp
    import optax
    hp = helpers["params"]
    loss_fn = helpers["loss_vtrace_lambda_sequence"]
    graphs = (helpers["lambda1_critic_graph"], helpers["lambda2_critic_graph"])
    def loss(q, target, graph, lam, batch):
        tm = lambda k: jnp.swapaxes(batch[k], 0, 1)
        return loss_fn(state.actor_target, q, target, graph, lam,
            tm("actions"), tm("rewards"), tm("terminated"), tm("behaviour_weight"),
            tm("latent"), tm("target_latent"), jax.random.PRNGKey(0))[0]
    @partial(jax.jit, static_argnames=("steps",))
    def train_kernel(heads, key, rate, data, steps):
        optimizer = optax.adam(rate)
        def update(carry, _):
            h, key = carry
            key, sample_key = jax.random.split(key)
            indices = jax.random.randint(sample_key, (batch_size,), 0, data["latent"].shape[0])
            batch = jax.tree.map(lambda a: a[indices], data)
            l1, g1 = jax.value_and_grad(loss)(h.q1, h.target1, graphs[0], hp.lambda1, batch)
            l2, g2 = jax.value_and_grad(loss)(h.q2, h.target2, graphs[1], hp.lambda2, batch)
            u1, o1 = optimizer.update(g1, h.opt1)
            u2, o2 = optimizer.update(g2, h.opt2)
            q1, q2 = optax.apply_updates(h.q1, u1), optax.apply_updates(h.q2, u2)
            tau = jnp.where((h.step + 1) % hp.periodic_update == 0, hp.tau, 0.)
            ema = lambda a, b: jax.tree.map(lambda x, y: (1 - tau) * x + tau * y, a, b)
            nh = Heads(q1, q2, ema(h.target1, q1), ema(h.target2, q2), o1, o2, h.step + 1)
            return (nh, key), jnp.stack((l1, l2))
        (h, key), losses = jax.lax.scan(update, (heads, key), None, length=steps)
        return h, key, losses.mean(0)
    def train(heads, key, rate, steps):
        # Pass large frozen caches as runtime device inputs, never XLA constants.
        return train_kernel(heads, key, rate, dataset, steps)
    train.kernel = train_kernel
    @jax.jit
    def measure(heads, batch, reference):
        x = batch["latent"].reshape(-1, batch["latent"].shape[-1])
        # Exclude the same trace tail as the production loss before ranking.
        keep = batch["latent"].shape[1] - hp.lambda_truncation
        qs, refs = [], []
        for q, ref, graph in zip((heads.q1, heads.q2), (reference.q1, reference.q2), graphs):
            shape = (*batch["latent"].shape[:2], -1, 2)
            qs.append(helpers["run_critic"](q, graph, x).reshape(shape)[:, :keep])
            refs.append(helpers["run_critic"](ref, graph, x).reshape(shape)[:, :keep])
        a, b = (q.min(-1) for q in qs)
        ar, br = (q.min(-1) for q in refs)
        actions = batch["actions"][:, :keep, 0].astype(jnp.int32)
        take = lambda q: jnp.take_along_axis(q, actions[..., None], axis=-1)[..., 0]
        d, dr = take(a - b), take(ar - br)
        return {"lambda1_bellman_mse": loss(heads.q1, heads.target1, graphs[0], hp.lambda1, batch),
                "lambda2_bellman_mse": loss(heads.q2, heads.target2, graphs[1], hp.lambda2, batch),
                "discrepancy_rms_taken": jnp.sqrt(jnp.mean(d ** 2)),
                "discrepancy_vector_drift_rms": jnp.sqrt(jnp.mean((d - dr) ** 2)),
                "lambda1_q_drift_rms": jnp.sqrt(jnp.mean((a - ar) ** 2)),
                "lambda2_q_drift_rms": jnp.sqrt(jnp.mean((b - br) ** 2)),
                "lambda1_argmax_change": jnp.mean(jnp.argmax(a, -1) != jnp.argmax(ar, -1)),
                "lambda2_argmax_change": jnp.mean(jnp.argmax(b, -1) != jnp.argmax(br, -1)),
                "lambda1_twin_gap": jnp.mean(jnp.abs(qs[0][..., 0] - qs[0][..., 1])),
                "lambda2_twin_gap": jnp.mean(jnp.abs(qs[1][..., 0] - qs[1][..., 1]))}
    return train, measure


def make_gradient_probe(state, debug, helpers, raw, config):
    """Direction of the *FE* discrepancy gradient, never applied to memory."""
    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    raw = jax.tree.map(jnp.asarray, raw)
    carry_dim = 0 if config["memory_type"] == "identity" else config["memory_hidden_dim"] * (2 if config["memory_type"] == "lstm" else 1)
    def objective(fe, q1, q2):
        x, _ = debug.calculate_latent(fe, state.feature_extractor_target,
            raw["observations"], raw["actions"], raw["terminated"],
            jnp.zeros((len(raw["observations"]), carry_dim), jnp.float32))
        # Exact production discrepancy term uses all post-burn-in steps but last.
        actions = jnp.swapaxes(raw["actions"][:, config["burn_in_length"]:], 0, 1)[:-1]
        return helpers["loss_ld"](q1, q2, helpers["lambda1_critic_graph"], helpers["lambda2_critic_graph"],
            x[:-1].reshape(-1, x.shape[-1]), actions.reshape(-1, 1))[0]
    @jax.jit
    def probe(heads):
        value, gradient = jax.value_and_grad(objective)(state.feature_extractor, heads.q1, heads.q2)
        vector, _ = ravel_pytree(gradient)
        return value, vector
    return probe


def array_hash(tree):
    import jax
    import numpy as np
    h = hashlib.sha256()
    h.update(str(jax.tree.structure(tree)).encode())
    for a in jax.tree.leaves(tree):
        a = np.asarray(a)
        h.update(str((a.shape, a.dtype)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def make_aux_evaluator(port, fns, debug, config):
    """New fixed evaluation episodes; no transitions enter the fitting dataset."""
    import jax
    import jax.numpy as jnp
    carry_dim = 0 if config["memory_type"] == "identity" else config["memory_hidden_dim"] * (2 if config["memory_type"] == "lstm" else 1)
    @partial(jax.jit, static_argnames=("mode",))
    def evaluate(state, seed, mode):
        def episode(index):
            key = jax.random.PRNGKey(seed + index)
            key, reset_key = jax.random.split(key)
            obs, es = port.reset(reset_key)
            def step(carry, _):
                key, obs, es, memory, prev, active = carry
                key, ek = jax.random.split(key)
                values, memory = debug.predict_values(state, obs, memory, prev)
                action = jnp.argmax(values[mode].min(-1)).astype(jnp.int32)
                obs, es, reward, done = port.transition(ek, es, action)
                return (key, obs, es, memory, fns.encode_action(jnp.atleast_1d(action)), active & ~done), (jnp.where(active, reward, 0.), active, active & done)
            initial = (key, obs, es, jnp.zeros(carry_dim, jnp.float32), jnp.zeros(port.action_dim, jnp.float32), jnp.bool_(True))
            _, outputs = jax.lax.scan(step, initial, None, length=port.horizon)
            return outputs
        return jax.vmap(episode)(jnp.arange(100, dtype=jnp.uint32))
    return evaluate


def evaluate_aux(evaluator, state, seed):
    import jax
    import numpy as np
    results = {}
    for mode in ("lambda1_q", "lambda2_q"):
        rewards, active, done = jax.device_get(evaluator(state, np.uint32(seed + 7000000), mode))
        if not done.any(1).all():
            raise RuntimeError("Auxiliary-greedy evaluation did not terminate naturally")
        returns = rewards.sum(1)
        if not np.isfinite(returns).all():
            raise FloatingPointError("Nonfinite auxiliary-greedy return")
        results[mode] = {"return_mean": float(returns.mean()), "return_sd_episodes": float(returns.std(ddof=1)),
                         "episode_returns": returns.tolist(), "length_mean": float(active.sum(1).mean()), "episodes": 100}
    return results


def worker(root, index):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_jax import PortEnv
    plan = verify(root)
    job = plan["jobs"][index]
    folder = root / "runs" / job["env_id"] / f"seed{job['seed']}"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "status.json", {"state": "running", "job": index, "pid": os.getpid()})
    source = Path(job["checkpoint"])
    saved = cloudpickle.loads(source.read_bytes())
    config = saved["config"]
    if config != job["config"] or saved["steps"] != 250000:
        raise ValueError("Checkpoint configuration changed")
    port = PortEnv(job["env_id"])
    built, fns, debug = build_agent(config, port)
    del built
    state = jax.tree.map(jnp.asarray, saved["state"])
    frozen_before = array_hash(state)
    helpers = production_helpers(fns)
    window = config["burn_in_length"] + config["sequence_length"] + config["lambda_truncation"]
    selections, split = split_replay(saved["state"].online_buffer, window, plan["train_windows"], plan["holdout_windows"], job["seed"] + 5000000)
    raw_train = raw_windows(saved["state"].online_buffer, selections["train"])
    raw_test = raw_windows(saved["state"].online_buffer, selections["holdout"])
    np.savez_compressed(folder / "window-indices.npz", **selections)
    write_json(folder / "dataset.json", split)
    print("CACHE", job["env_id"], job["seed"], flush=True)
    dataset = cache_features(state, debug, raw_train, config)
    holdout = cache_features(state, debug, raw_test, config)
    # Fixed measurement batches; holdout is never used for optimization.
    validation = jax.tree.map(lambda a: a[:128], holdout)
    training_eval = jax.tree.map(lambda a: a[:128], dataset)
    probe = make_gradient_probe(state, debug, helpers, {k: v[:8] for k, v in raw_test.items()}, config)
    reference = initial_heads(state)
    base_value, base_gradient = jax.device_get(probe(reference))
    base_norm = np.linalg.norm(base_gradient)
    train, measure = make_refitter(state, helpers, dataset, config["batch_size"])
    evaluator = make_aux_evaluator(port, fns, debug, config)
    result = {"env_id": job["env_id"], "seed": job["seed"], "source": job,
              "dataset": split, "frozen_snapshot_sha256": frozen_before, "branches": {},
              "initial_aux_greedy": evaluate_aux(evaluator, state, job["seed"])}
    start = time.monotonic()
    for rate in plan["learning_rates"]:
        run = None
        if plan["wandb"]:
            import wandb
            run = wandb.init(project="lambda-popgym-critic-settling", entity="fhstp-data-intelligence-research-group",
                id=hashlib.sha256(f"{folder.resolve()}:{rate}".encode()).hexdigest()[:16], resume="allow",
                name=f"{job['env_id']}-lambda-seed{job['seed']}-lr{rate}", group=job["env_id"],
                config={**job, "learning_rate": rate, "protocol": plan["protocol"]}, dir=str(folder))
            run.define_metric("updates")
            run.define_metric("diagnostic/*", step_metric="updates")
        heads, key, previous = reference, jax.random.PRNGKey(job["seed"] + 6000000), 0
        history = []
        for steps in plan["milestones"]:
            if steps > previous:
                # Same start RNG and minibatch stream in both LR branches.
                remaining = steps - previous
                while remaining:
                    chunk = min(plan["update_chunk"], remaining)
                    heads, key, losses = jax.block_until_ready(train(heads, key, rate, chunk))
                    if not np.isfinite(np.asarray(losses)).all():
                        raise FloatingPointError("Nonfinite training loss")
                    remaining -= chunk
            rows = {name: {k: float(v) for k, v in jax.device_get(measure(heads, data, reference)).items()}
                    for name, data in (("train", training_eval), ("holdout", validation))}
            value, gradient = jax.device_get(probe(heads))
            norm = np.linalg.norm(gradient)
            cosine = float(np.dot(base_gradient, gradient) / (base_norm * norm)) if base_norm > 0 and norm > 0 else None
            row = {"updates": steps, **rows, "fe_discrepancy_loss": float(value), "fe_gradient_norm": float(norm),
                   "fe_gradient_cosine_to_initial": cosine, "elapsed_seconds": time.monotonic() - start}
            if not all(np.isfinite(v) for vals in rows.values() for v in vals.values()) or not np.isfinite(norm):
                raise FloatingPointError("Nonfinite diagnostic")
            history.append(row)
            result["branches"][str(rate)] = history
            write_json(folder / "results.json", result)
            if run:
                run.log({"updates": steps,
                    **{f"diagnostic/{label}/{k}": v for label, vals in rows.items() for k, v in vals.items()},
                    "diagnostic/fe_gradient_norm": float(norm), "diagnostic/fe_gradient_cosine": cosine})
            print("MEASURE", job["env_id"], job["seed"], rate, steps, json.dumps(row), flush=True)
            previous = steps
        refit_state = with_heads(state, heads)
        # Verify every non-auxiliary state field exactly; no actor/memory update.
        allowed = {"lambda1_critic", "lambda2_critic", "lambda1_critic_target", "lambda2_critic_target",
                   "lambda1_critic_optimizer_state", "lambda2_critic_optimizer_state"}
        untouched = tuple(getattr(refit_state, k) for k in state._fields if k not in allowed)
        original = tuple(getattr(state, k) for k in state._fields if k not in allowed)
        if array_hash(untouched) != array_hash(original):
            raise AssertionError("Frozen state mutated")
        # Save separate heads, never overwrite source checkpoint.
        temp = folder / f"heads-lr{rate}.pkl.tmp"
        with temp.open("wb") as file:
            cloudpickle.dump(jax.device_get(heads), file)
        temp.replace(folder / f"heads-lr{rate}.pkl")
        result.setdefault("freeze_checks", {})[str(rate)] = True
        final_aux = evaluate_aux(evaluator, refit_state, job["seed"])
        result.setdefault("final_aux_greedy", {})[str(rate)] = final_aux
        write_json(folder / "results.json", result)
        if run:
            for mode, vals in final_aux.items():
                run.summary[f"final_aux_greedy/{mode}/return_mean"] = vals["return_mean"]
            run.summary["freeze_checks_passed"] = True
            run.finish()
    if array_hash(state) != frozen_before or file_hash(source) != job["checkpoint_sha256"]:
        raise AssertionError("Original checkpoint or state mutated")
    write_json(folder / "status.json", {"state": "complete", "updates_per_branch": plan["milestones"][-1]})


def prepare(root, source_root, gpus):
    from popgym_suite import source_fingerprint
    from popgym_parallel import current_port_fingerprint
    import importlib.metadata
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Use a separate diagnostic root")
    jobs = []
    for env in ENVS:
        for seed in range(42, 47):
            folder = source_root / env / "lambda" / f"seed{seed}"
            provenance = json.loads((folder / "config.json").read_text())
            if provenance["source_sha256"] != source_fingerprint():
                raise ValueError("Training source mismatch")
            status = json.loads((folder / "status.json").read_text())
            cfg = provenance["experiment"]
            if status["state"] != "complete" or status["steps"] != 250000 or cfg["backend"] != "jax" or cfg["alpha"] != .1:
                raise ValueError("Need completed matched alpha=.1 JAX snapshots")
            checkpoint = folder / "checkpoint.pkl"
            jobs.append({"env_id": env, "seed": seed, "checkpoint": str(checkpoint.resolve()),
                         "checkpoint_sha256": file_hash(checkpoint), "config": cfg})
    jobs.sort(key=lambda job: (job["seed"], ENVS.index(job["env_id"])))
    plan = {"created_at": datetime.now().astimezone().isoformat(), "jobs": jobs, "gpus": gpus,
            "learning_rates": [1e-4, 1e-5], "milestones": list(MILESTONES), "train_windows": 2048,
            "update_chunk": 250,
            "execution": "Feature banks passed as runtime GPU inputs, fixed 250-update compiled chunks. Separate rerun after first implementation's host OOM; no budget, dataset, optimizer or loss changes.",
            "holdout_windows": 512, "source_sha256": source_fingerprint(), "script_sha256": file_hash(__file__),
            "wandb": True,
            "port_sha256": current_port_fingerprint(), "versions": {n: importlib.metadata.version(n) for n in ("jax", "flax", "optax", "popjym")},
            "protocol": "Two environments, five paired seeds, same fixed replay bank and RNG per LR. Original Adam moments and lambda targets preserved. Only lambda heads and their EMA targets update. Online and target actor/FE copies remain frozen exactly as in the snapshot. No environment data collection, FE update or oracle claim.",
            "holdout": "Last ~20% chronological replay region, split at a terminal; windows share no source slots with training. Not IID: replay was collected during online learning. Evaluation uses 128 fixed windows; FE gradient uses 8 fixed holdout windows. No checkpoint selection by holdout. Auxiliary-greedy endpoint evaluation: 100 fixed new episodes at seed+7M, identical before/after, never used for fitting.",
            "resume_sweep_on_success": SWEEP_UNIT}
    write_json(root / "plan.json", plan)
    return plan


def verify(root):
    from popgym_suite import source_fingerprint
    from popgym_parallel import current_port_fingerprint
    import importlib.metadata
    plan = json.loads((root / "plan.json").read_text())
    if plan["script_sha256"] != file_hash(__file__) or plan["source_sha256"] != source_fingerprint() or plan["port_sha256"] != current_port_fingerprint():
        raise ValueError("Diagnostic or production sources changed")
    if any(importlib.metadata.version(n) != v for n, v in plan["versions"].items()):
        raise ValueError("Package versions changed")
    for job in plan["jobs"]:
        if file_hash(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("Source checkpoint changed")
    return plan


def render(root, web):
    import statistics
    web.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    parts = ['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Frozen critic settling</title>',
        '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.6rem;border:1px solid #ccc}th{background:#eee}</style>',
        '<h1>Frozen-policy / frozen-memory critic settling</h1><p><a href="/alpha-sweep/">Alpha sweep</a> · <a href="/">Weekend results</a></p>',
        '<p>Only auxiliary critics are fitted. Two tasks × five seeds × two learning rates. Same fixed replay windows, initial weights/moments, and minibatch RNG. Source checkpoints are read-only.</p>',
        '<p>Held-out Bellman MSE is not oracle Q-value error. Discrepancy drift and FE-gradient changes diagnose sensitivity to critic fitting; they do not establish correctness or improved memory. Main critic and actor action rankings cannot change in this experiment.</p>',
        '<table><tr><th>Task / LR</th><th>Updates</th><th>Seeds</th><th>Holdout λ1 MSE</th><th>Holdout λ2 MSE</th><th>Discrepancy RMS</th><th>Discrepancy drift</th><th>FE gradient cosine</th><th>λ1 / λ2 ranking change</th></tr>']
    grouped = {}
    for p in (root / "runs").glob("*/seed*/results.json"):
        r = json.loads(p.read_text())
        for rate, history in r["branches"].items():
            for row in history:
                grouped.setdefault((r["env_id"], float(rate), row["updates"]), []).append(row)
    summary = []
    for (env, rate, steps), rows in sorted(grouped.items()):
        mean = lambda field: statistics.fmean(row["holdout"][field] for row in rows)
        cos = [r["fe_gradient_cosine_to_initial"] for r in rows if r["fe_gradient_cosine_to_initial"] is not None]
        fields = ("lambda1_bellman_mse", "lambda2_bellman_mse", "discrepancy_rms_taken", "discrepancy_vector_drift_rms")
        metrics = {k: mean(k) for k in fields}
        cosine = statistics.fmean(cos) if cos else None
        summary.append({"env_id": env, "learning_rate": rate, "updates": steps, "seeds": len(rows), **metrics, "fe_gradient_cosine_mean": cosine})
        cells = [html.escape(env) + f' / {rate:g}', str(steps), str(len(rows)),
                 *(f'{metrics[k]:.5g}' for k in fields), f'{cosine:.3f}' if cosine is not None else 'undefined',
                 f'{mean("lambda1_argmax_change"):.3f} / {mean("lambda2_argmax_change"):.3f}']
        parts.append('<tr>' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>')
    parts.append('</table><h2>Run status</h2><ul>')
    for job in plan["jobs"]:
        p = root / "runs" / job["env_id"] / f"seed{job['seed']}" / "status.json"
        state = json.loads(p.read_text()).get("state") if p.exists() else 'queued'
        parts.append(f'<li>{html.escape(job["env_id"])} seed{job["seed"]}: {html.escape(state)}</li>')
    parts.append('</ul><p>Counts are training seeds, not windows. Fixed chronological holdout is distribution-shifted, not an independent environment-level generalization test. Full per-seed trajectories and source hashes are retained.</p>')
    temp = web / 'index.html.tmp'
    temp.write_text(''.join(parts)); temp.replace(web / 'index.html')
    write_json(root / 'summary.json', summary)


def supervise(root):
    plan = verify(root)
    queue = []
    for i, job in enumerate(plan["jobs"]):
        path = root / "runs" / job["env_id"] / f"seed{job['seed']}" / "status.json"
        if not path.exists() or json.loads(path.read_text()).get("state") != "complete":
            queue.append(i)
    slots, active, failures = list(plan["gpus"]), [], []
    (root / "logs").mkdir(exist_ok=True)
    web = Path('/tmp/opencode/popgym-report-web/critic-settling')
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while queue or active:
            while slots and queue:
                index, gpu = queue.pop(0), slots.pop(0)
                log = (root / "logs" / f"{index}.log").open('a')
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "JAX_PLATFORMS": "cuda",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"}
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', str(index)],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((p, gpu, index, log))
                print('START', index, gpu, flush=True)
            for item in list(active):
                p, gpu, index, log = item
                code = p.poll()
                if code is None:
                    continue
                log.close(); active.remove(item); slots.append(gpu)
                if code:
                    failures.append(index)
                    job = plan['jobs'][index]
                    folder = root / 'runs' / job['env_id'] / f"seed{job['seed']}"
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / 'status.json', {'state': 'failed', 'exit_code': code})
                print('DONE' if not code else 'FAILED', index, flush=True)
            render(root, web)
            write_json(root / 'status.json', {'state': 'running', 'queued': queue, 'active': [i for _, _, i, _ in active], 'failures': failures})
            if active:
                time.sleep(10)
        verify(root)
        if failures:
            raise RuntimeError(f'Diagnostic jobs failed: {failures}; sweep remains paused for inspection')
        subprocess.run(['systemctl', '--user', 'thaw', SWEEP_UNIT], check=True)
        pause_path = Path('/tmp/opencode/popgym-alpha-followup/priority-pause.json')
        if pause_path.exists():
            record = json.loads(pause_path.read_text())
            write_json(pause_path, {**record, 'state': 'resumed', 'resumed_at': datetime.now().astimezone().isoformat()})
        write_json(root / 'status.json', {'state': 'complete', 'sweep_resumed': True, 'completed_at': datetime.now().astimezone().isoformat()})
    except BaseException as error:
        write_json(root / 'status.json', {'state': 'interrupted', 'error': repr(error),
                   'active_at_interruption': [i for _, _, i, _ in active], 'sweep_resumed': False,
                   'interrupted_at': datetime.now().astimezone().isoformat()})
        raise
    finally:
        for p, _, _, _ in active:
            p.terminate()
        for p, _, index, log in active:
            p.wait(); log.close()
            job = plan['jobs'][index]
            folder = root / 'runs' / job['env_id'] / f"seed{job['seed']}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'status.json'
            if not path.exists() or json.loads(path.read_text()).get('state') != 'complete':
                write_json(path, {'state': 'interrupted', 'exit_code': p.returncode})
        render(root, web)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('/tmp/opencode/popgym-alpha-followup/alpha0p1'))
    p.add_argument('--gpus', nargs='+', default=['0', '1', '2'])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--supervise', action='store_true')
    mode.add_argument('--worker', type=int)
    args = p.parse_args()
    if args.prepare:
        plan = prepare(args.root.resolve(), args.source_root.resolve(), args.gpus)
        print(f"Prepared {len(plan['jobs'])} paired-seed jobs, two LR branches each")
    elif args.worker is not None:
        worker(args.root.resolve(), args.worker)
    else:
        supervise(args.root.resolve())


if __name__ == '__main__':
    main()
