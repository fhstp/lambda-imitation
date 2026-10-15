"""JAX-only, isolated temperature diagnostics and fixed-alpha experiments.

Saved weekend policies are read-only. Every alpha (including 0.1) is trained
afresh with the same paired seeds/budget; no continuation or baseline splicing.
Uses the existing independent-seed runner without changing frozen algorithms.
"""

import argparse
from functools import partial
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


DEFAULT_ENVS = ["popgym-RepeatPreviousEasy-v0", "popgym-RepeatPreviousHard-v0",
                "popgym-HigherLowerHard-v0", "popgym-PositionOnlyCartPoleEasy-v0",
                "popgym-BattleshipEasy-v0"]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def alpha_name(alpha):
    return "alpha" + str(float(alpha)).replace(".", "p")


def tempered_probs(probs, temperature):
    import jax
    import jax.numpy as jnp
    # Evaluator accepts only positive temperatures here; T=0 uses argmax.
    log_probs = jnp.log(jnp.maximum(probs, jnp.finfo(probs.dtype).tiny))
    return jax.nn.softmax(log_probs / temperature)


def make_evaluator(port, fns, debug, carry_dim):
    """Evaluate actual executed-action histories, with resets and no training."""
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=("mode", "n_episodes"))
    def evaluate(state, seed, temperature, mode="actor", n_episodes=100):
        def episode(index):
            key = jax.random.PRNGKey(seed + index)
            key, rk = jax.random.split(key)
            obs, es = port.reset(rk)
            def step(c, _):
                key, obs, es, carry, prev, active = c
                key, ak, ek = jax.random.split(key, 3)
                values, nc = debug.predict_values(state, obs, carry, prev)
                probs, q = values["probs"], values["q"].min(-1)
                sampled = tempered_probs(probs, jnp.maximum(temperature, 1e-6))
                if mode == "critic":
                    action = jnp.argmax(q)
                    effective = jax.nn.one_hot(action, port.action_dim)
                elif mode == "random":
                    action = jax.random.randint(ak, (), 0, port.action_dim)
                    effective = jnp.ones_like(probs) / port.action_dim
                else:
                    greedy = jnp.argmax(probs)
                    action = jnp.where(temperature == 0., greedy,
                                       jax.random.categorical(ak, jnp.log(sampled)))
                    effective = jnp.where(temperature == 0., jax.nn.one_hot(greedy, port.action_dim), sampled)
                nobs, nes, reward, done = port.transition(ek, es, action)
                entropy = lambda p: -(p * jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny))).sum()
                sorted_q = jnp.sort(q)
                metrics = jnp.array([entropy(probs), entropy(effective), probs.max(), effective.max(),
                                     sorted_q[-1] - sorted_q[-2], (jnp.argmax(q) != jnp.argmax(probs)).astype(jnp.float32)])
                outputs = (jnp.where(active, reward, 0.), active, active & done,
                           jnp.where(active, metrics, jnp.zeros_like(metrics)))
                return (key, nobs, nes, nc, fns.encode_action(jnp.atleast_1d(action)), active & ~done), outputs
            initial = (key, obs, es, jnp.zeros(carry_dim, jnp.float32),
                       jnp.zeros(port.action_dim, jnp.float32), jnp.bool_(True))
            _, outputs = jax.lax.scan(step, initial, None, length=port.horizon)
            return outputs
        return jax.vmap(episode)(jnp.arange(n_episodes, dtype=jnp.uint32))
    return evaluate


def evaluation_summary(outputs, alpha):
    import numpy as np
    rewards, active, done, diagnostics = map(np.asarray, outputs)
    if not done.any(axis=1).all():
        raise RuntimeError("Evaluation failed to terminate naturally")
    returns = rewards.sum(axis=1)
    values = diagnostics.sum(axis=(0, 1)) / active.sum()
    names = ("actor_entropy", "executed_policy_entropy", "actor_max_probability",
             "executed_policy_max_probability", "critic_top2_gap", "actor_critic_argmax_disagreement")
    result = {"return_mean": float(returns.mean()), "return_sd_episodes": float(returns.std(ddof=1)) if len(returns) > 1 else 0.,
              "episodes": len(returns), "length_mean": float(active.sum(axis=1).mean()),
              **dict(zip(names, map(float, values)))}
    result["critic_top2_gap_over_alpha"] = result["critic_top2_gap"] / alpha
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite temperature evaluation")
    return {"metrics": result, "episode_returns": returns.tolist()}


def build_diagnostic_agent(args, method, seed, port):
    import jax.numpy as jnp
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import EnvSpec, create_iqlearn_from_env
    hp = Hyperparameters(batch_size=1, online_batch_size=args.batch_size, online_buffer_size=args.online_buffer_size,
                         fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, lambda_critic_lr=args.critic_lr,
                         alpha=args.alpha, gamma=args.gamma, tau=args.tau, burn_in_length=args.burn_in_length,
                         sequence_length=args.sequence_length, lambda_truncation=args.lambda_truncation,
                         lambda1=args.lambda1, lambda2=args.lambda2, lambda_coef=args.lambda_coef,
                         fake_onpolicy_loss=False, retrace=True, stop_actor_fe=True)
    _, fns, debug = create_iqlearn_from_env(EnvSpec(port.obs_shape, port.action_dim, True),
        {"observations": jnp.zeros((1, *port.obs_shape)), "actions": jnp.zeros((1, 1))},
        buffer_size=1, hp=hp, projection=args.projection_dim or None, memory_type=args.memory_type,
        memory_hidden_dim=args.memory_hidden_dim, use_prev_action=True, actor_dims=(args.head_dim,),
        critic_dims=(args.head_dim,), lambda1_critic_dims=(args.head_dim,), lambda2_critic_dims=(args.head_dim,),
        approximate_lambda=method == "lambda", critic_layer_norm=args.critic_layer_norm,
        debug=True, seed=seed, use_sac=args.use_sac)
    carry_dim = 0 if args.memory_type == "identity" else args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1)
    return make_evaluator(port, fns, debug, carry_dim)


def temperature_job(job, plan, root):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_jax import PortEnv
    from popgym_suite import parser, write_json
    port = PortEnv(job["env_id"])
    evaluator = None
    for item in job["checkpoints"]:
        source = Path(item["folder"])
        if file_hash(source / "checkpoint.pkl") != item["checkpoint_sha256"]:
            raise ValueError("Source checkpoint changed after planning")
        config = json.loads((source / "config.json").read_text())["experiment"]
        args = parser().parse_args([])
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
        saved = cloudpickle.loads((source / "checkpoint.pkl").read_bytes())
        if saved["config"] != config or saved["steps"] != item["steps"]:
            raise ValueError("Checkpoint/config mismatch")
        if evaluator is None:
            evaluator = build_diagnostic_agent(args, job["method"], item["seed"], port)
        state = jax.tree.map(jnp.asarray, saved["state"])
        result = {**item, "env_id": job["env_id"], "method": job["method"], "training_alpha": args.alpha,
                  "eval_seed": item["seed"] + 3_000_000, "evaluation_kind": "new fixed diagnostic cases, not original final holdout",
                  "temperatures": {}, "references": {}}
        for temperature in plan["temperatures"]:
            outputs = jax.device_get(evaluator(state, np.uint32(result["eval_seed"]), temperature, "actor", plan["temperature_episodes"]))
            result["temperatures"][str(temperature)] = evaluation_summary(outputs, args.alpha)
        for mode in ("critic", "random"):
            outputs = jax.device_get(evaluator(state, np.uint32(result["eval_seed"]), 1., mode, plan["temperature_episodes"]))
            result["references"][mode] = evaluation_summary(outputs, args.alpha)
        folder = root / "temperature" / job["env_id"] / job["method"] / f"seed{item['seed']}"
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "results.json", result)
        if plan["wandb"]:
            import wandb
            run = wandb.init(project=plan["wandb_project"], entity=plan["wandb_entity"],
                id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16], resume="allow",
                name=f"temperature-{job['env_id']}-{job['method']}-{item['seed']}",
                group="posthoc-temperature", config={k: v for k, v in result.items() if k not in ("temperatures", "references")}, dir=str(folder))
            for t, summary in result["temperatures"].items():
                run.log({"temperature": float(t), **{f"diagnostic/{k}": v for k, v in summary["metrics"].items()}})
            run.finish()
        print("TEMPERATURE", job["env_id"], job["method"], item["seed"],
              {t: round(v["metrics"]["return_mean"], 4) for t, v in result["temperatures"].items()}, flush=True)
        del saved, state, outputs


def verify_plan(plan):
    from popgym_parallel import current_port_fingerprint, runner_fingerprint
    from popgym_suite import source_fingerprint
    if plan["sources"] != source_fingerprint() or plan["runner_sha256"] != runner_fingerprint() or plan["followup_sha256"] != file_hash(__file__):
        raise ValueError("Follow-up sources changed; refusing mixed experiments")
    if plan["port_sha256"] != current_port_fingerprint():
        raise ValueError("JAX environment sources changed")
    if any(importlib.metadata.version(n) != v for n, v in plan["versions"].items()):
        raise ValueError("Package versions changed")


def prepare(options, args):
    from popgym_jax import assign_backends
    from popgym_parallel import current_port_fingerprint, runner_fingerprint
    from popgym_suite import discover, source_fingerprint, write_json
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        verify_plan(plan)
        if plan["suite_args"] != vars_for_json(args) or plan["alphas"] != options.alphas or plan["temperatures"] != options.temperatures or plan["temperature_episodes"] != options.temperature_episodes or plan["checkpoint_root"] != str(options.checkpoint_root.resolve()):
            raise ValueError("Follow-up settings changed; use a separate output directory")
        return plan
    included, excluded = discover(args.envs or DEFAULT_ENVS)
    included = assign_backends(included, "jax")  # unsupported tasks fail; no fallback
    if excluded or any(e["backend"] != "jax" for e in included):
        raise ValueError("Follow-ups require discrete audited JAX environments only")
    old = json.loads((options.checkpoint_root / "manifest.json").read_text())
    old_tasks = old["tasks"]
    sources = source_fingerprint()
    jobs, missing = [], []
    for env in included:
        for method in args.methods:
            checkpoints = []
            for task in old_tasks:
                if (task["env_id"], task["method"]) != (env["env_id"], method):
                    continue
                folder = options.checkpoint_root.resolve() / task["env_id"] / method / f"seed{task['seed']}"
                status_path = folder / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                if status.get("state") != "complete":
                    continue
                provenance = json.loads((folder / "config.json").read_text())
                if provenance["source_sha256"] != sources or task["backend"] != "jax":
                    raise ValueError("Saved policy is not from the unchanged audited JAX suite")
                checkpoints.append({"folder": str(folder), "seed": task["seed"], "steps": status["steps"],
                                    "checkpoint_sha256": file_hash(folder / "checkpoint.pkl")})
            if checkpoints:
                jobs.append({"kind": "temperature", "env_id": env["env_id"], "method": method, "checkpoints": checkpoints})
            else:
                missing.append({"env_id": env["env_id"], "method": method, "reason": "no completed JAX checkpoint"})
    for alpha in options.alphas:
        alpha_root = root / alpha_name(alpha)
        alpha_root.mkdir(parents=True, exist_ok=True)
        tasks = [{"env_id": env["env_id"], "backend": "jax", "port_id": env["port_id"],
                  "backend_note": env["backend_note"], "expected_source_sha256": sources,
                  "method": method, "seed": args.seed + i, "target_steps": args.steps}
                 for env in included for method in args.methods for i in range(args.num_seeds)]
        write_json(alpha_root / "manifest.json", {"included": included, "excluded": [], "tasks": tasks,
                   "note": f"Fresh paired-seed alpha sweep, fixed alpha={alpha}; JAX-only; original algorithms/budgets unchanged"})
        for env in included:
            for method in args.methods:
                group = [t for t in tasks if t["env_id"] == env["env_id"] and t["method"] == method]
                jobs.append({"kind": "train", "alpha": alpha, "root": str(alpha_root), "tasks": group})
    plan = {"suite_args": vars_for_json(args), "checkpoint_root": str(options.checkpoint_root.resolve()),
            "alphas": options.alphas, "temperatures": options.temperatures, "temperature_episodes": options.temperature_episodes,
            "sources": sources, "runner_sha256": runner_fingerprint(), "followup_sha256": file_hash(__file__),
            "port_sha256": current_port_fingerprint(), "versions": {n: importlib.metadata.version(n) for n in ("popjym", "gymnax", "jax", "flax", "optax", "cloudpickle")},
            "wandb": args.wandb, "wandb_project": args.wandb_project, "wandb_entity": args.wandb_entity,
            "jobs": jobs, "missing_checkpoints": missing,
            "execution": "Native-precision independent-seed vmap; fresh training for each alpha, including baseline. Intentional new alpha experiment, not continuation of certified weekend suite.",
            "evaluation": "Posthoc temperatures use seed+3M; training validation uses seed+1M, final holdout seed+2M. Same fixed cases across candidates. No temperature selected/retrained from original holdout."}
    write_json(plan_path, plan)
    return plan


def vars_for_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def args_from_plan(plan):
    from popgym_suite import parser
    args = parser().parse_args([])
    for key, value in plan["suite_args"].items():
        setattr(args, key, Path(value) if key in ("output_dir", "task_file") and value is not None else value)
    return args


def worker(root, index):
    from popgym_suite import write_json
    plan = json.loads((root / "plan.json").read_text())
    verify_plan(plan)
    job = plan["jobs"][index]
    status = root / "jobs" / f"{index}.json"
    write_json(status, {"state": "running", "job": job, "pid": os.getpid()})
    try:
        if job["kind"] == "temperature":
            temperature_job(job, plan, root)
        else:
            from popgym_parallel import train_group
            args = args_from_plan(plan)
            args.alpha, args.output_dir, args.resume = job["alpha"], Path(job["root"]), True
            train_group(args, job["tasks"], plan["runner_sha256"])
        write_json(status, {"state": "complete", "job": job})
    except BaseException:
        import traceback
        write_json(status, {"state": "failed", "job": job, "error": traceback.format_exc()})
        raise


def supervise(options, args):
    from popgym_suite import write_json
    root = args.output_dir.resolve()
    plan = prepare(options, args)
    (root / "jobs").mkdir(exist_ok=True)
    queue = []
    for i in range(len(plan["jobs"])):
        path = root / "jobs" / f"{i}.json"
        if not path.exists() or json.loads(path.read_text()).get("state") != "complete":
            queue.append(i)
    slots = list(args.gpus or [])
    if not slots:
        raise ValueError("Specify --gpus for JAX follow-ups")
    active, failures = [], []
    print(f"FOLLOWUP {len(queue)} jobs; {len(slots)} GPUs; JAX-only; root={root}", flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while queue or active:
            verify_plan(plan)
            while queue and slots:
                i, gpu = queue.pop(0), slots.pop(0)
                log = (root / "jobs" / f"{i}.log").open("a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "JAX_PLATFORMS": "cuda",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONUNBUFFERED": "1"}
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(i),
                                            "--output-dir", str(root)], stdout=log, stderr=subprocess.STDOUT, env=env)
                active.append((process, gpu, i, log))
                print("START", i, gpu, plan["jobs"][i]["kind"], flush=True)
            for item in list(active):
                process, gpu, i, log = item
                code = process.poll()
                if code is None:
                    continue
                log.close(); active.remove(item); slots.append(gpu)
                if code:
                    failures.append(i)
                    write_json(root / "jobs" / f"{i}.json", {"state": "failed", "exit_code": code, "job": plan["jobs"][i]})
                print("DONE" if code == 0 else "FAILED", i, "remaining", len(queue), flush=True)
            if active:
                time.sleep(2)
    finally:
        for process, _, _, _ in active:
            process.terminate()
        for process, _, _, log in active:
            process.wait(); log.close()
    if failures:
        raise SystemExit(f"Follow-up jobs failed: {failures}; saved checkpoints retained")


def render_report(root, web):
    from popgym_report import build, collect, CSS, esc, finals, stats
    from popgym_suite import write_json
    plan = json.loads((root / "plan.json").read_text())
    web.mkdir(parents=True, exist_ok=True)
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>POPGym alpha sweep</title><style>{CSS}</style></head><body><header><h1>JAX-only · Temperature diagnostics &amp; α sweep</h1>',
             '<p><a href="/">Weekend results (stopped broad queue)</a></p>',
             '<p>Fresh training at each fixed α, including 0.1; paired seeds and budgets. Both methods retain reward-only critic targets. Diagnostic sampling temperature is not training α. Missing/partial results are not zeros; seed means, not pooled episodes, are aggregated.</p>',
             '<h2>Training α sweep · final holdout means</h2><div class="table-wrap"><table><thead><tr><th>Environment / method / α</th><th>Greedy actor</th><th>Stochastic actor</th><th>Critic-greedy</th><th>Progress</th></tr></thead><tbody>']
    for alpha in plan["alphas"]:
        ar = root / alpha_name(alpha)
        snapshot = collect(ar)
        build(ar, web / f"{alpha_name(alpha)}.html", 60)
        parts.append(f'<tr><td colspan="5"><a href="{alpha_name(alpha)}.html">α={alpha} · learning curves and detailed seed coverage</a></td></tr>')
        for env_id, env in sorted(snapshot["environments"].items()):
            for method, runs in sorted(env["methods"].items()):
                parts.append(f'<tr><td>{esc(env_id)} / {esc(method)} / {alpha}</td>')
                for mode in ("greedy", "stochastic", "critic"):
                    results = finals(runs, mode)
                    text = 'Pending' if not results else '<br>'.join(f'{r["mean"]:.3f} ± {r["sd"]:.3f} (n={r["n"]})' if r["sd"] is not None else f'{r["mean"]:.3f} (n={r["n"]}; SD unavailable)' for _, r in results)
                    parts.append(f'<td>{text}</td>')
                parts.append(f'<td>{sum(r["complete"] for r in runs)}/{len(runs)} complete; saved steps {min(r["saved_steps"] for r in runs):,}–{max(r["saved_steps"] for r in runs):,}</td></tr>')
    parts.append('</tbody></table></div><h2>Saved-policy sampling temperatures · new diagnostic cases</h2><p>T=1 is the original stochastic actor; T=0 is greedy. Every temperature uses the same episode keys per seed. These are descriptive diagnostics, not a fresh training-seed replication or an independent confirmation of a selected temperature.</p><div class="table-wrap"><table><thead><tr><th>Environment / method / T</th><th>Return ± seed SD</th><th>Seeds</th><th>Original actor H</th><th>Executed H</th><th>Original max π</th><th>Executed max π</th><th>Q gap / α</th><th>Argmax disagreement</th></tr></thead><tbody>')
    grouped = {}
    for file in sorted((root / "temperature").glob("*/*/seed*/results.json")):
        result = json.loads(file.read_text())
        for t, data in result["temperatures"].items():
            grouped.setdefault((result["env_id"], result["method"], float(t)), []).append(data["metrics"])
        for mode, data in result["references"].items():
            grouped.setdefault((result["env_id"], result["method"], mode), []).append(data["metrics"])
    aggregates = []
    for (env, method, t), rows in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        summary = stats([r["return_mean"] for r in rows])
        text = f'{summary["mean"]:.3f}' + (f' ± {summary["sd"]:.3f}' if summary["sd"] is not None else ' (SD unavailable)')
        metrics = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        aggregates.append({"env_id": env, "method": method, "temperature_or_reference": t, "return": summary, "metrics_mean_across_seeds": metrics})
        parts.append(f'<tr><td>{esc(env)} / {esc(method)} / {esc(t)}</td><td>{text}</td><td>{len(rows)}</td>')
        for name in ("actor_entropy", "executed_policy_entropy", "actor_max_probability", "executed_policy_max_probability", "critic_top2_gap_over_alpha", "actor_critic_argmax_disagreement"):
            parts.append(f'<td>{metrics[name]:.4f}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div><p>Only completed weekend checkpoints are evaluated, so diagnostic seed coverage differs by task. All source policies were trained at α=0.1; checkpoint hashes and episode returns are retained. Q gap is the top-two minimum-twin critic gap, not necessarily the correct-vs-wrong gap. No published PPO scores are treated as a matched comparison.</p></header></body></html>')
    temp = web / "index.html.tmp"
    temp.write_text(''.join(parts)); temp.replace(web / "index.html")
    write_json(root / "temperature-summary.json", aggregates)


def main():
    from popgym_suite import parser, validate_args
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--checkpoint-root", type=Path, default=Path("/tmp/opencode/popgym-weekend"))
    p.add_argument("--alphas", nargs="+", type=float, default=[.1, .01, .001])
    p.add_argument("--temperatures", nargs="+", type=float, default=[1., .3, .1, .03, 0.])
    p.add_argument("--temperature-episodes", type=int, default=100)
    p.add_argument("--report", type=Path)
    p.add_argument("--watch", type=int, default=0)
    options, remaining = p.parse_known_args()
    args = parser().parse_args(remaining)
    validate_args(args)
    if any(not math.isfinite(a) or a <= 0 for a in options.alphas) or len(set(options.alphas)) != len(options.alphas):
        p.error("Need unique positive alphas")
    if any(not math.isfinite(t) or t < 0 for t in options.temperatures) or len(set(options.temperatures)) != len(options.temperatures) or options.temperature_episodes <= 0:
        p.error("Need unique nonnegative temperatures and positive episode count")
    if options.worker is not None:
        worker(args.output_dir.resolve(), options.worker)
    elif options.report:
        while True:
            render_report(args.output_dir.resolve(), options.report)
            if not options.watch:
                break
            time.sleep(options.watch)
    else:
        if args.backend != "jax":
            p.error("Explicit --backend jax required; no Gymnasium training fallback")
        if options.prepare_only:
            plan = prepare(options, args)
            print(f'Prepared {len(plan["jobs"])} jobs in {args.output_dir}', flush=True)
        else:
            supervise(options, args)


if __name__ == "__main__":
    main()
