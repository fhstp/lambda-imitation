"""Isolated seed batching for the frozen POPGym suite.

The original single-seed runner and its source guards remain unchanged.
Benchmarking writes only to a separate directory, never to suite checkpoints.
"""

import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def stack(trees):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def unstack(tree):
    import jax
    size = jax.tree.leaves(tree)[0].shape[0]
    return [jax.tree.map(lambda x: x[i], tree) for i in range(size)]


class SeedRunner:
    """One unchanged algorithm, independent state/replay/keys per seed."""

    def __init__(self, runner, precision=None):
        import jax

        @partial(jax.jit, static_argnames=("n_steps",))
        def train(states, rollouts, n_steps):
            # Native precision, as in the existing scalar runner. Different
            # CUDA kernels need not produce matching floating-point trajectories.
            with jax.default_matmul_precision(precision):
                return jax.vmap(lambda s, r: runner.train(s, r, n_steps))(states, rollouts)

        @partial(jax.jit, static_argnames=("n_steps",))
        def prefill(states, rollouts, n_steps):
            return jax.vmap(lambda s, r: runner.prefill(s, r, n_steps))(states, rollouts)

        self.train, self.prefill = train, prefill


def initialize(args, task, port):
    """Mirror train_one_jax initialization; use its original graph/functions."""
    import jax.numpy as jnp
    from lambda_imitation.buffer import create_buffer
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import EnvSpec, create_iqlearn_from_env
    from popgym_jax import JaxRunner

    hp = Hyperparameters(batch_size=1, online_batch_size=args.batch_size, online_buffer_size=args.online_buffer_size,
                         fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, lambda_critic_lr=args.critic_lr,
                         alpha=args.alpha, gamma=args.gamma, tau=args.tau, burn_in_length=args.burn_in_length,
                         sequence_length=args.sequence_length, lambda_truncation=args.lambda_truncation,
                         lambda1=args.lambda1, lambda2=args.lambda2, lambda_coef=args.lambda_coef,
                         fake_onpolicy_loss=False, retrace=True, stop_actor_fe=True)
    approximate_lambda = task["method"] == "lambda"
    state, fns, debug = create_iqlearn_from_env(
        EnvSpec(port.obs_shape, port.action_dim, True),
        {"observations": jnp.zeros((1, *port.obs_shape)), "actions": jnp.zeros((1, 1))},
        buffer_size=1, hp=hp, projection=args.projection_dim or None, memory_type=args.memory_type,
        memory_hidden_dim=args.memory_hidden_dim, use_prev_action=True, actor_dims=(args.head_dim,),
        critic_dims=(args.head_dim,), lambda1_critic_dims=(args.head_dim,), lambda2_critic_dims=(args.head_dim,),
        approximate_lambda=approximate_lambda, critic_layer_norm=args.critic_layer_norm,
        debug=True, seed=task["seed"], use_sac=args.use_sac)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
    carry_dim = 0 if args.memory_type == "identity" else args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1)
    runner = JaxRunner(args, port, fns, debug, bfns, carry_dim, approximate_lambda)
    return state, runner.initial_rollout(task["seed"]), runner


def compare_trees(reference, actual, rtol=1e-4, atol=1e-6):
    """Integer/RNG/action leaves exact, floats tolerate batched GEMM roundoff."""
    import jax
    import numpy as np
    if jax.tree.structure(reference) != jax.tree.structure(actual):
        raise AssertionError("Different state structures")
    maximum, failures = 0., []
    for (path, ref), value in zip(jax.tree_util.tree_flatten_with_path(reference)[0], jax.tree.leaves(actual)):
        ref, value = np.asarray(ref), np.asarray(value)
        if ref.shape != value.shape:
            raise AssertionError(f"Different shapes at {path}")
        if np.issubdtype(ref.dtype, np.inexact):
            if ref.size:
                maximum = max(maximum, float(np.max(np.abs(ref - value))))
            if not np.allclose(ref, value, rtol=rtol, atol=atol):
                failures.append(str(path))
        elif not np.array_equal(ref, value):
            failures.append(str(path))
    return {"passed": not failures, "max_abs_float_difference": maximum,
            "rtol": rtol, "atol": atol, "failed_leaves": failures}


def benchmark(args, options):
    import jax
    import numpy as np
    from popgym_jax import PortEnv, port_fingerprint
    from popgym_suite import run_config, source_fingerprint, write_json

    options.output.mkdir(parents=True, exist_ok=True)
    report = {"devices": [str(d) for d in jax.devices()], "source_sha256": source_fingerprint(),
              "batch_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {n: importlib.metadata.version(n) for n in ("popjym", "gymnax", "jax", "flax", "optax", "cloudpickle")},
              "port_source_sha256": port_fingerprint(), "cases": []}
    write_json(options.output / "benchmark.json", report)
    for env_id in options.envs:
        port = PortEnv(env_id)
        for method in options.methods:
            print(f"START {env_id}/{method} with {options.seeds} independent seeds", flush=True)
            task = {"env_id": env_id, "method": method, "seed": args.seed, "backend": "jax"}
            start = time.monotonic()
            states, rollouts, runner = [], [], None
            for i in range(options.seeds):
                state, rollout, built = initialize(args, {**task, "seed": args.seed + i}, port)
                states.append(state); rollouts.append(rollout)
                runner = runner or built
            batch = SeedRunner(runner)
            n_prefill = args.prefill_steps or args.batch_size * (args.burn_in_length + args.sequence_length + args.lambda_truncation)
            bs, br, interactions = jax.block_until_ready(batch.prefill(stack(states), stack(rollouts), n_prefill))
            single_states, single_rollouts = unstack(bs), unstack(br)
            # Both paths begin with identical independent per-seed state.
            check_steps = args.collect_steps
            ref = [jax.block_until_ready(runner.train(s, r, check_steps)) for s, r in zip(single_states, single_rollouts)]
            actual = jax.block_until_ready(batch.train(bs, br, check_steps))
            correctness = compare_trees(stack(ref), actual)
            case = {"env_id": env_id, "method": method, "seeds": options.seeds,
                    "experiment": run_config(args, task), "prefill_interactions": np.asarray(interactions).tolist(),
                    "correctness": correctness, "setup_and_check_seconds": time.monotonic() - start,
                    "rounds": [], "n_steps_per_seed_per_round": options.chunk_steps,
                    "vmap_matmul_precision": "native"}
            print("CORRECTNESS", json.dumps(correctness), flush=True)
            # Compile full-size calls before timing. No compilation time is
            # credited as a steady-state speedup; report it separately.
            compile_start = time.monotonic()
            scalar = runner.train.lower(single_states[0], single_rollouts[0], options.chunk_steps).compile()
            vector = batch.train.lower(bs, br, options.chunk_steps).compile()
            case["full_chunk_compile_seconds"] = time.monotonic() - compile_start
            # Warm up both execution paths, then alternate order to mitigate
            # concurrent-suite load/thermal effects on the shared GPU.
            jax.block_until_ready(scalar(single_states[0], single_rollouts[0]))
            jax.block_until_ready(vector(bs, br))
            for repeat in range(options.repeats):
                times = {}
                for name in (("single", "vmap") if repeat % 2 == 0 else ("vmap", "single")):
                    start = time.monotonic()
                    if name == "single":
                        outs = [scalar(s, r) for s, r in zip(single_states, single_rollouts)]
                        jax.block_until_ready(outs)
                    else:
                        jax.block_until_ready(vector(bs, br))
                    times[name] = time.monotonic() - start
                case["rounds"].append(times)
                print("TIMING", env_id, method, repeat, json.dumps(times), flush=True)
            scalar_seconds = float(np.median([r["single"] for r in case["rounds"]]))
            vector_seconds = float(np.median([r["vmap"] for r in case["rounds"]]))
            case.update(single_median_seconds=scalar_seconds, vmap_median_seconds=vector_seconds,
                        aggregate_speedup=scalar_seconds / vector_seconds,
                        single_aggregate_steps_per_second=options.seeds * options.chunk_steps / scalar_seconds,
                        vmap_aggregate_steps_per_second=options.seeds * options.chunk_steps / vector_seconds)
            report["cases"].append(case)
            write_json(options.output / "benchmark.json", report)
            print("RESULT", json.dumps({k: case[k] for k in ("env_id", "method", "aggregate_speedup", "vmap_aggregate_steps_per_second")}), flush=True)
            # Bound memory between differently shaped environment compilations.
            del scalar, vector, ref, actual, bs, br, states, rollouts, single_states, single_rollouts, batch, runner, built, state, rollout, outs
            jax.clear_caches()
    return report


def runner_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def current_port_fingerprint():
    """Same hash as popgym_jax.port_fingerprint without importing JAX."""
    import importlib.util
    root = Path(importlib.util.find_spec("popjym").origin).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def verify_sources(tasks, expected_runner):
    from popgym_suite import source_fingerprint
    if expected_runner and runner_fingerprint() != expected_runner:
        raise ValueError("Batch runner changed since scheduling/benchmark; refusing mixed sources")
    sources = source_fingerprint()
    for task in tasks:
        if task["expected_source_sha256"] != sources:
            raise ValueError("Original suite sources changed; refusing mixed implementations")


def train_group(args, tasks, expected_runner):
    """Save/log the original per-seed schema, with separate execution provenance."""
    import cloudpickle
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_jax import PortEnv, evaluation_result, port_fingerprint
    from popgym_suite import run_config, source_fingerprint, write_json

    verify_sources(tasks, expected_runner)
    ledger = args.output_dir / "parallel-execution.json"
    if ledger.exists():
        evidence = json.loads(ledger.read_text())["benchmark"]
        if evidence["port_source_sha256"] != current_port_fingerprint():
            raise ValueError("POPJym sources changed since benchmark")
        if any(importlib.metadata.version(n) != version for n, version in evidence["versions"].items()):
            raise ValueError("Packages changed since scheduling")
    identities = {(t["env_id"], t["method"], t["backend"], t["target_steps"]) for t in tasks}
    if len(identities) != 1 or tasks[0]["backend"] != "jax":
        raise ValueError("Seed groups require the same JAX environment, method and budget")
    if tasks[0]["target_steps"] != args.steps:
        raise ValueError("CLI training budget differs from frozen task budget")
    if len({t["seed"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate seed in batch")
    port = PortEnv(tasks[0]["env_id"])
    versions = {n: importlib.metadata.version(n) for n in ("popjym", "gymnax", "jax", "flax", "optax", "cloudpickle")}
    lanes, states, rollouts, runner = [], [], [], None
    # Check every lane before touching status/checkpoint files or opening W&B.
    for task in tasks:
        folder = args.output_dir / task["env_id"] / task["method"] / f"seed{task['seed']}"
        folder.mkdir(parents=True, exist_ok=True)
        config = run_config(args, task)
        config.update(actual_obs_shape=list(port.obs_shape), actual_action_dim=port.action_dim,
                      episode_horizon=port.horizon, port_source_sha256=port_fingerprint())
        provenance = {"experiment": config, "source_sha256": source_fingerprint(), "versions": versions}
        config_path, checkpoint = folder / "config.json", folder / "checkpoint.pkl"
        if config_path.exists():
            old = json.loads(config_path.read_text())
            if any(old[k] != value for k, value in provenance.items()):
                raise ValueError(f"Existing run has different settings/source/packages: {folder}")
            if not args.resume:
                raise FileExistsError(f"Existing run requires --resume: {folder}")
        elif checkpoint.exists():
            raise ValueError(f"Checkpoint without source/config record: {folder}")
        saved = cloudpickle.loads(checkpoint.read_bytes()) if checkpoint.exists() else None
        if saved and saved["config"] != config:
            raise ValueError(f"Checkpoint config mismatch: {folder}")
        steps = saved["steps"] if saved else 0
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if steps >= args.steps and status.get("state") == "complete":
            continue
        state, rollout, built = initialize(args, task, port)
        runner = runner or built
        if saved:
            state, rollout = jax.tree.map(jnp.asarray, (saved["state"], saved["rollout"]))
        states.append(state); rollouts.append(rollout)
        lanes.append({"task": task, "folder": folder, "config": config, "provenance": provenance,
                      "steps": steps, "start_steps": steps,
                      "interactions": saved["interactions"] if saved else 0,
                      "history": saved["history"] if saved else [], "run": None})
    if not lanes:
        return
    if len({lane["steps"] for lane in lanes}) != 1 or len({lane["interactions"] == 0 for lane in lanes}) != 1:
        # An interruption between checkpoint and status writes can make the
        # supervisor's status-based cohort stale. Split by actual checkpoints,
        # rather than discard progress or advance some lanes extra steps.
        cohorts = {}
        for lane in lanes:
            cohorts.setdefault((lane["steps"], lane["interactions"] == 0), []).append(lane["task"])
        for cohort in cohorts.values():
            train_group(args, cohort, expected_runner)
        return
    batch = SeedRunner(runner)
    states, rollouts = stack(states), stack(rollouts)
    start, steps = time.monotonic(), lanes[0]["steps"]
    approximate_lambda = tasks[0]["method"] == "lambda"

    def status(lane, state, **extra):
        write_json(lane["folder"] / "status.json", {"state": state, "steps": lane["steps"],
                   "target_steps": args.steps, **extra})

    def save(lane, state, rollout):
        data = {"config": lane["config"], "state": jax.device_get(state), "rollout": jax.device_get(rollout),
                "steps": lane["steps"], "interactions": lane["interactions"], "history": lane["history"]}
        checkpoint = lane["folder"] / "checkpoint.pkl"
        tmp = checkpoint.with_suffix(".pkl.tmp")
        with tmp.open("wb") as file:
            cloudpickle.dump(data, file)
        tmp.replace(checkpoint)
        write_json(lane["folder"] / "history.json", lane["history"])

    def evaluate(final=False):
        n = args.final_episodes if final else args.eval_episodes
        modes = ("greedy", "stochastic", "critic", "random") if final else ("greedy", "stochastic", "critic")
        for lane, state, rollout in zip(lanes, unstack(states), unstack(rollouts)):
            if final and lane["history"] and lane["history"][-1].get("final") and lane["history"][-1]["steps"] == steps:
                continue
            seed = np.uint32((lane["task"]["seed"] + (2_000_000 if final else 1_000_000)) % 2**32)
            results = {m: evaluation_result(jax.device_get(runner.evaluate(state, seed, m, n)), m, args, approximate_lambda) for m in modes}
            elapsed = time.monotonic() - start
            lane["history"].append({"steps": steps, "interactions_including_prefill": lane["interactions"],
                "final": final, "backend": "jax", "elapsed_seconds_this_process": elapsed,
                "steps_this_process": steps - lane["start_steps"], "execution": "seed_vmap",
                "eval": {m: r["metrics"] for m, r in results.items()}})
            write_json(lane["folder"] / f"episodes_{steps}{'_final' if final else ''}.json", results)
            if lane["run"]:
                prefix = "final" if final else "eval"
                lane["run"].log({"steps": steps, **{f"{prefix}/{m}/{k}": v for m, r in results.items() for k, v in r["metrics"].items()}})
            save(lane, state, rollout)
            status(lane, "running")
            print(json.dumps({**lane["task"], "steps": steps, "final": final, "execution": "seed_vmap",
                              "returns": {m: r["metrics"]["return_mean"] for m, r in results.items()}}), flush=True)

    try:
        for lane in lanes:
            folder = lane["folder"]
            if not (folder / "config.json").exists():
                write_json(folder / "config.json", {**lane["provenance"], "devices": [str(d) for d in jax.devices()]})
            write_json(folder / "execution.json", {"kind": "seed_vmap", "runner_sha256": runner_fingerprint(),
                        "original_source_sha256": source_fingerprint(), "group_seeds": [l["task"]["seed"] for l in lanes],
                        "start_steps": steps, "matmul_precision": "native",
                        "note": "Original experiment/checkpoint schema unchanged; independent seed state; CUDA roundoff may change training trajectories"})
            if args.wandb:
                import wandb
                task = lane["task"]
                lane["run"] = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16], resume="allow", reinit="create_new",
                    group=task["env_id"], name=f"{task['env_id']}-{task['method']}-{task['seed']}",
                    config=lane["config"], dir=str(folder))
                lane["run"].define_metric("steps")
                for prefix in ("train", "eval", "final"):
                    lane["run"].define_metric(f"{prefix}/*", step_metric="steps")
            status(lane, "running")
        if lanes[0]["interactions"] == 0:
            n = args.prefill_steps or args.batch_size * (args.burn_in_length + args.sequence_length + args.lambda_truncation)
            states, rollouts, counts = jax.block_until_ready(batch.prefill(states, rollouts, n))
            if np.any(np.asarray(rollouts[-1]) != 0) or np.any(np.asarray(rollouts[-2])):
                raise RuntimeError("Batched random prefill did not reach natural episode boundaries")
            for lane, count in zip(lanes, np.asarray(counts)):
                lane["interactions"] = int(count)
            evaluate()
        while steps < args.steps:
            boundary = min(args.steps, (steps // args.eval_every + 1) * args.eval_every)
            count = min(args.jax_train_steps, boundary - steps)
            states, rollouts, metrics = batch.train(states, rollouts, count)
            metrics = jax.device_get(metrics)
            if not all(np.isfinite(v).all() for v in metrics.values()):
                raise FloatingPointError("Nonfinite batched training metrics")
            if np.any(np.asarray(rollouts[-2])):
                raise RuntimeError("Batched training episode exceeded safety horizon")
            steps += count
            for lane in lanes:
                lane["steps"] = steps
                lane["interactions"] += count
            if steps == boundary:
                for i, lane in enumerate(lanes):
                    if lane["run"]:
                        lane["run"].log({"steps": steps, **{f"train/{k}": float(v[i]) for k, v in metrics.items()}})
                evaluate()
        # If an interruption saved final evaluation before complete status, do
        # not append/log that evaluation twice on resume.
        needs_final = [not l["history"] or not l["history"][-1]["final"] or l["history"][-1]["steps"] != steps for l in lanes]
        if any(needs_final):
            evaluate(final=True)
        for lane in lanes:
            status(lane, "complete", interactions_including_prefill=lane["interactions"])
    except BaseException:
        for lane in lanes:
            status(lane, "failed", error=traceback.format_exc())
        raise
    finally:
        for lane in lanes:
            if lane["run"]:
                lane["run"].finish()


def plan_groups(root, tasks, batch_seeds):
    """Environment-first; completed work is skipped without spawning workers."""
    from popgym_report import sort_key
    grouped = {}
    for task in tasks:
        folder = root / task["env_id"] / task["method"] / f"seed{task['seed']}"
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("state") == "complete" and status.get("steps") == task["target_steps"]:
            # Require both checkpoint and final holdout before skipping a task.
            history_path = folder / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            if (folder / "checkpoint.pkl").exists() and any(r.get("final") and r.get("steps") == task["target_steps"] for r in history):
                continue
        steps = status.get("steps", 0) if (folder / "checkpoint.pkl").exists() else 0
        if task["backend"] == "jax":
            grouped.setdefault((task["env_id"], task["method"], steps), []).append(task)
        else:
            grouped[(task["env_id"], task["method"], task["seed"])] = [task]
    queue = []
    for _, group in sorted(grouped.items(), key=lambda kv: (sort_key(kv[0][0]), kv[0][1], kv[0][2])):
        group.sort(key=lambda task: task["seed"])
        size = batch_seeds if group[0]["backend"] == "jax" else 1
        queue.extend(group[i:i + size] for i in range(0, len(group), size))
    return queue


def verify_benchmark(gate, tasks, expected_runner, args, min_speedup=1.1):
    """No change of execution without passing full-budget per-method evidence."""
    from popgym_suite import run_config, source_fingerprint
    data = json.loads(Path(gate).read_text())
    if data.get("validation_complete") is not True:
        raise ValueError("Scientific validation has not finished")
    if data["source_sha256"] != source_fingerprint() or data["batch_runner_sha256"] != expected_runner:
        raise ValueError("Benchmark certificate does not match current algorithm/batch sources")
    if any(importlib.metadata.version(n) != version for n, version in data["versions"].items()):
        raise ValueError("Packages changed since benchmark")
    if data["port_source_sha256"] != current_port_fingerprint():
        raise ValueError("POPJym sources changed since benchmark")
    validation = data.get("scientific_validation", {})
    validation_path = Path(__file__).with_name("popgym_parallel_validation.py")
    if validation.get("source_sha256") != hashlib.sha256(validation_path.read_bytes()).hexdigest():
        raise ValueError("Scientific validation missing or validation source changed")
    audited = {r["env_id"] for r in validation.get("environments", []) if r.get("passed")}
    if not {t["env_id"] for t in tasks if t["backend"] == "jax"} <= audited:
        raise ValueError("Missing full-episode environment/reset validation")
    methods = {t["method"] for t in tasks if t["backend"] == "jax"}
    for method in methods:
        cases = [c for c in data["cases"] if c["method"] == method]
        required = ("repeatable", "resume_exact", "seed_isolation", "finite", "learning_sanity")
        if not cases or any(c["aggregate_speedup"] < min_speedup or
                            any(c.get("safety", {}).get(k) is not True for k in required) for c in cases):
            raise ValueError(f"No passing speed/correctness benchmark for {method}; keep single-seed queue")
        expected = run_config(args, next(t for t in tasks if t["method"] == method and t["backend"] == "jax"))
        task_fields = set(tasks[0])
        for case in cases:
            if any(case["experiment"].get(k) != v for k, v in expected.items() if k not in task_fields):
                raise ValueError(f"Benchmark used different algorithm settings for {method}")
    return data


def supervise(args, options, remaining):
    """Original manifest and worker CLI, changed scheduling/execution only."""
    from popgym_suite import summarize, write_json
    root = args.output_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = manifest["tasks"]
    expected_runner = options.runner_sha256 or runner_fingerprint()
    verify_sources(tasks, expected_runner)
    if any(t["target_steps"] != args.steps for t in tasks):
        raise ValueError("CLI training budget differs from frozen manifest")
    evidence = verify_benchmark(options.benchmark_gate, tasks, expected_runner, args)
    # Scheduling-only audit trail; never replace the experiment manifest.
    write_json(root / "parallel-execution.json", {"runner_sha256": expected_runner,
               "benchmark_file": str(options.benchmark_gate.resolve()), "benchmark": evidence,
               "schedule": "environment-first, equal-checkpoint-step seed groups", "max_batch_seeds": options.batch_seeds})
    queue = plan_groups(root, tasks, options.batch_seeds)
    slots = [gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)] if args.gpus else [None] * args.workers
    active, failures = [], 0
    task_dir = root / "seed-groups"
    task_dir.mkdir(exist_ok=True)
    print(f"{sum(map(len, queue))} unfinished runs in {len(queue)} groups; {len(slots)} workers; environment-first", flush=True)
    last_progress = time.monotonic()
    try:
        while queue or active:
            for gpu in list(slots):
                if not queue:
                    break
                group = queue.pop(0)
                task = group[0]
                name = f"{task['env_id']}_{task['method']}_" + "-".join(str(t["seed"]) for t in group)
                file = task_dir / f"{name}.json"
                write_json(file, group)
                log_path = task_dir / f"{name}.log"
                log = log_path.open("a")
                environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"}
                if gpu is not None:
                    environment.update(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cuda")
                if task["backend"] == "jax" and len(group) > 1:
                    command = [sys.executable, str(Path(__file__).resolve()), "--group-file", str(file),
                               "--runner-sha256", expected_runner, *remaining]
                else:
                    single_file = task_dir / f"{name}_single.json"
                    write_json(single_file, task)
                    command = [sys.executable, str(Path(__file__).with_name("popgym_suite.py")), *remaining,
                               "--task-file", str(single_file)]
                process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                active.append((process, gpu, log, group, log_path))
                slots.remove(gpu)
                print(f"START gpu={gpu} {name}; log={log_path}", flush=True)
            for item in list(active):
                process, gpu, log, group, log_path = item
                code = process.poll()
                if code is None:
                    continue
                log.close(); active.remove(item); slots.append(gpu)
                if code:
                    failures += len(group)
                    for task in group:
                        folder = root / task["env_id"] / task["method"] / f"seed{task['seed']}"
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / "status.json"
                        saved = json.loads(path.read_text()) if path.exists() else {}
                        # Keep saved steps: subsequent resume can group correctly.
                        write_json(path, {**saved, "state": "failed", "exit_code": code, "error": f"See {log_path}"})
                print(f"{'FAILED' if code else 'DONE'} {group[0]['env_id']}/{group[0]['method']}/seeds={[t['seed'] for t in group]}; {len(queue)} groups queued", flush=True)
                summarize(root, tasks)
            if active:
                if time.monotonic() - last_progress >= 60:
                    print(f"ACTIVE {[(g[0]['env_id'], g[0]['method'], [t['seed'] for t in g]) for _, _, _, g, _ in active]}; {len(queue)} groups queued", flush=True)
                    last_progress = time.monotonic()
                time.sleep(1)
    except KeyboardInterrupt:
        for process, _, _, _, _ in active:
            process.terminate()
        for process, _, log, _, _ in active:
            process.wait(); log.close()
        raise
    finally:
        summarize(root, tasks)
    if args.plot:
        from popgym_suite import plot_curves
        plot_curves(root, tasks)
    if failures:
        raise SystemExit(f"{failures} runs failed; checkpoints retained; inspect seed-groups/*.log")


def main():
    from popgym_suite import parser, validate_args
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--parallel-supervise", action="store_true")
    mode.add_argument("--group-file", type=Path)
    p.add_argument("--envs", nargs="+", default=["popgym-BattleshipEasy-v0"])
    p.add_argument("--methods", nargs="+", choices=("lambda", "sac"), default=["lambda", "sac"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--chunk-steps", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-seeds", type=int, default=5)
    p.add_argument("--runner-sha256")
    p.add_argument("--benchmark-gate", type=Path)
    options, remaining = p.parse_known_args()
    if min(options.seeds, options.chunk_steps, options.repeats, options.batch_seeds) <= 0:
        p.error("Seeds, chunk steps and repeats must be positive")
    args = parser().parse_args(remaining)
    validate_args(args)
    if options.benchmark:
        if options.output is None:
            p.error("--benchmark requires --output")
        benchmark(args, options)
    elif options.group_file:
        train_group(args, json.loads(options.group_file.read_text()), options.runner_sha256)
    else:
        if options.benchmark_gate is None or not args.resume:
            p.error("--parallel-supervise requires --benchmark-gate and --resume")
        supervise(args, options, remaining)


if __name__ == "__main__":
    main()
