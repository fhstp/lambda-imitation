"""One fresh scientific job per tmux worker. No training resume/retry option."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from .learner import create_learner, paper_settings
from .native_validation import digest_tree, validate_fixed_dataset
from .records import (enqueue_upload, load_owned_pickle, read, save_pickle, sha, utc,
                      verify_sources, write)


ROOT = Path(__file__).resolve().parents[2]


def summarize(values, prefix, seeds):
    result = {}
    for name, values in values.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(seeds),) or not np.isfinite(array).all():
            raise FloatingPointError("nonfinite/missing seed metric: "+prefix+"/"+name)
        name = "return" if name == "return_" else name
        result.update({f"{prefix}/{name}/seed_{i}": float(v) for i, v in enumerate(array)})
        result[f"{prefix}/{name}/mean"] = float(array.mean())
        result[f"{prefix}/{name}/std"] = float(array.std(ddof=1))
    return result


def finite_networks(agent, metrics):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((agent[:15], metrics)))


def build_batch(settings, seeds):
    built = [create_learner(settings, seed) for seed in seeds]
    agent = jax.tree.map(lambda *x: jnp.stack(x), *[x[0] for x in built])
    rollout = jax.tree.map(lambda *x: jnp.stack(x), *[x[1] for x in built])
    return agent, rollout, built[0][2]


def eval_keys(seeds, round_, stream):
    return jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.key(s), stream), round_) for s in seeds])


def execute_job(state, job, directory, study, admission, tracking):
    seeds = job["seeds"]
    environment = job["environment"]
    settings = paper_settings(environment, measure=job["measure"], correction=job["correction"],
                              frozen=job["kind"] == "fit")
    agent, rollout, fns = build_batch(settings, seeds)
    history = []
    metrics_file = directory/"metrics.jsonl"
    if metrics_file.exists(): raise FileExistsError("attempt already has metrics; no resumption")

    def log(row):
        history.append(row)
        with metrics_file.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        if tracking:
            try: tracking.log(row)
            except Exception as e:
                with (directory/"administrative-errors.jsonl").open("a") as stream:
                    stream.write(json.dumps({"utc": utc(), "operation": "wandb.log", "error": repr(e)})+"\n")
        progress = {k:v for k,v in row.items() if k in ("round","updates","env_interactions",
            "timing/train_seconds","eval/greedy/return/mean","fixed_mu_heldout/discrepancy_rms/mean",
            "dataset_prepared","physical_calls_per_seed","train/importance_ratio_max/mean",
            "train/trace_coefficient_max/mean")}
        print(json.dumps(progress, allow_nan=False), flush=True)

    def checkpoint(name, payload, *, failure=False):
        path = directory/"checkpoints"/(name+".pkl.gz")
        record = save_pickle(path, jax.device_get(payload))
        enqueue_upload(directory, path, kind="checkpoint", release=not failure)
        return record

    def guard():
        verify_sources(ROOT, admission)
        if shutil.disk_usage(directory).free < study.get("disk_reserve_bytes", 100*2**30):
            raise RuntimeError("persistent disk reserve reached; no deletion or hidden retry")

    config = {"job": job, "settings": asdict(settings), "source_manifest_sha256": sha(state/"admission.json"),
              "training_resume_allowed": False, "starting_measure": job["measure"],
              "collector": "fixed-uniform" if job["kind"] != "learning" else "adaptive-online",
              "stationary_mu_claim": job["kind"] != "learning", "seeds": seeds,
              "runtime_devices": [str(x) for x in jax.devices()],
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    write(directory/"config.json", config, exclusive=True)
    enqueue_upload(directory, directory/"config.json", kind="config")
    if tracking: tracking.config.update(config)

    if job["kind"] == "learning":
        budget = study["learning"][environment]
        prefill = jax.jit(jax.vmap(lambda a, r: fns.prefill(a, r, steps=budget["prefill"])))
        write(directory/"production-started.json", {"utc": utc(), "resume": False}, exclusive=True)
        agent, rollout = prefill(agent, rollout)
        jax.block_until_ready(agent.update_step)
        fixed_prefill = validate_fixed_dataset(agent, environment)
        write(directory/"prefill-validation.json", {"actual_calls_per_seed": budget["prefill"],
                                                     "checks": fixed_prefill}, exclusive=True)
        train = jax.jit(jax.vmap(lambda a, r, k: fns.train(a, r, k, steps=budget["steps_per_round"])))
        keys = eval_keys(seeds, 0, 0x7A11)
        evaluators = {mode: jax.jit(jax.vmap(lambda a, k, greedy=(mode=="greedy"):
            fns.evaluate(a, k, episodes=budget["eval_episodes"], greedy=greedy))) for mode in ("greedy", "sampled")}

        def evaluate(round_):
            values = {}
            for mode, function in evaluators.items():
                values.update(summarize(function(agent, eval_keys(seeds, round_, 0xE001)), f"eval/{mode}", seeds))
            return values

        if budget["eval_initial"]:
            log({"round": 0, "env_interactions": budget["prefill"], **evaluate(0)})
        for round_ in range(1, budget["rounds"]+1):
            guard()
            started = time.perf_counter()
            agent, rollout, keys, metrics = train(agent, rollout, keys)
            jax.block_until_ready(agent.update_step)
            elapsed = time.perf_counter()-started
            calls = budget["prefill"] + round_*budget["steps_per_round"]
            if not finite_networks(agent, metrics):
                checkpoint(f"failure-round-{round_}", {"agent": agent, "rollout": rollout,
                    "keys": keys, "metrics": metrics, "history": history}, failure=True)
                raise FloatingPointError("nonfinite network/metric; attempt preserved, no clipping/retry")
            if not np.all(np.asarray(rollout.calls) == calls): raise RuntimeError("physical-call counter mismatch")
            if not np.all(np.asarray(agent.update_step) == round_*budget["steps_per_round"]):
                raise RuntimeError("learner update counter mismatch")
            row = {"round": round_, "env_interactions": calls, "timing/train_seconds": elapsed,
                   "timing/includes_compilation": round_ == 1, **summarize(metrics, "train", seeds),
                   **evaluate(round_)}
            log(row)
            if round_ == budget["reference_round"] and job["measure"] == "uniform" and job["correction"] == "retrace":
                path = directory/"reference.pkl.gz"
                record = save_pickle(path, {"networks": jax.device_get(agent[:3]), "seeds": seeds,
                    "settings": asdict(settings), "physical_calls": calls, "round": round_,
                    "source_admission_sha256": sha(state/"admission.json")})
                write(directory/"reference.json", {**record, "round": round_, "physical_calls": calls}, exclusive=True)
                enqueue_upload(directory, path, kind="frozen-reference")
            if round_ % 10 == 0 or round_ == budget["rounds"]:
                checkpoint(f"round-{round_:03d}", {"agent": agent, "rollout": rollout, "keys": keys,
                    "round": round_, "history": history, "config": config})
            write(directory/"status.json", {"status": "running", "kind": job["kind"], "round": round_,
                "env_interactions": calls, "utc": utc()})
        write(directory/"training-exit.json", {"exit_code": 0, "utc": utc(), "continuous_no_resume": True}, exclusive=True)
        summary = {"env_interactions": calls, "round": budget["rounds"],
                   "primary_endpoint": budget["primary_endpoint"], "all_seeds_retained": True}

    elif job["kind"] == "data":
        reference_dir = state/"jobs"/job["reference_job"]
        reference_record = read(reference_dir/"reference.json")
        reference = load_owned_pickle(reference_dir/reference_record["name"], reference_record["sha256"])
        if reference["seeds"] != seeds or reference["source_admission_sha256"] != sha(state/"admission.json"):
            raise RuntimeError("reference seed/source mismatch")
        budget = study["frozen"]
        # Uniform collection never consults the trained actor. Separate fold-in
        # keys make fitting/held-out data independent of the learning trajectory.
        from .collection import start_rollout
        from lambda_imitation.envs.battleship import Battleship
        from lambda_imitation.envs.minesweeper import MineSweeper
        env = Battleship() if environment == "battleship" else MineSweeper()
        shape = env.observation_space(env.default_params).shape
        max_steps = env.num_actions if environment == "battleship" else env.episode_length
        datasets, reports = {}, {}
        for name, calls_, stream in (("fitting", budget["data_calls_per_seed"], 0xDA71),
                                      ("heldout", budget["heldout_calls_per_seed"], 0xDA72)):
            fresh_agent, _, _ = build_batch(settings, seeds)
            fresh_rollout = jax.tree.map(lambda *x: jnp.stack(x), *[
                start_rollout(env, jax.random.fold_in(jax.random.key(s), stream), shape, max_steps) for s in seeds])
            prefill = jax.jit(jax.vmap(lambda a, r: fns.prefill(a, r, steps=calls_)))
            fresh_agent, fresh_rollout = prefill(fresh_agent, fresh_rollout)
            jax.block_until_ready(fresh_agent.update_step)
            reports[name] = validate_fixed_dataset(fresh_agent, environment)
            assert np.all(np.asarray(fresh_rollout.calls) == calls_)
            datasets[name] = jax.device_get(fresh_agent.online_buffer)
            reports[name] = {"checks": reports[name], "physical_calls_per_seed": calls_,
                             "pending_calls_per_seed": np.asarray(fresh_rollout.age).tolist()}
        path = directory/"fixed-datasets.pkl.gz"
        record = save_pickle(path, {**datasets, "seeds": seeds, "reference_sha256": reference_record["sha256"],
                                   "collector": "fixed-uniform", "reports": reports})
        write(directory/"datasets.json", {**record, "reference_job": job["reference_job"],
                                           "reference_sha256": reference_record["sha256"], "reports": reports}, exclusive=True)
        enqueue_upload(directory, path, kind="fixed-datasets")
        log({"dataset_prepared": True, "physical_calls_per_seed": budget["data_calls_per_seed"]+budget["heldout_calls_per_seed"]})
        summary = {"fixed_collector_verified": True, "reports": reports}

    elif job["kind"] == "fit":
        from .frozen_evaluation import create_evaluator
        data_dir = state/"jobs"/job["data_job"]
        data_record = read(data_dir/"datasets.json")
        data = load_owned_pickle(data_dir/data_record["name"], data_record["sha256"])
        reference_dir = state/"jobs"/data_record["reference_job"]
        reference_record = read(reference_dir/"reference.json")
        reference = load_owned_pickle(reference_dir/reference_record["name"], reference_record["sha256"])
        if reference_record["sha256"] != data["reference_sha256"] or reference["seeds"] != seeds or data["seeds"] != seeds or reference["source_admission_sha256"] != sha(state/"admission.json"):
            raise RuntimeError("frozen reference/data identity mismatch")
        networks = jax.tree.map(jnp.asarray, reference["networks"])
        agent = agent._replace(feature_extractor=networks[0], actor=networks[1], critic=networks[2],
            feature_extractor_target=networks[0], actor_target=networks[1], critic_target=networks[2],
            online_buffer=jax.tree.map(jnp.asarray, data["fitting"]))
        heldout = jax.tree.map(jnp.asarray, data["heldout"])
        frozen_hash = digest_tree(agent[:3])
        budget = study["frozen"]
        evaluator = create_evaluator(settings, fns)
        fit = jax.jit(jax.vmap(lambda a, k: fns.fit(a, k, steps=budget["eval_every_updates"])))
        keys = eval_keys(seeds, 0, 0xF171)
        write(directory/"production-started.json", {"utc": utc(), "resume": False,
            "frozen_reference_sha256": reference_record["sha256"], "dataset_sha256": data_record["sha256"]}, exclusive=True)
        for done in range(0, budget["updates"]+1, budget["eval_every_updates"]):
            if done:
                guard()
                agent, keys, metrics = fit(agent, keys)
                jax.block_until_ready(agent.update_step)
                if not finite_networks(agent, metrics):
                    checkpoint(f"failure-updates-{done}", {"agent": agent, "keys": keys, "metrics": metrics}, failure=True)
                    raise FloatingPointError("nonfinite frozen fit; preserve all seed outcomes")
                if digest_tree(agent[:3]) != frozen_hash:
                    raise RuntimeError("fixed encoder/actor/task critic changed")
            else:
                metrics = {}
            scored = evaluator(agent, heldout)
            arrays = {name: np.array([r[name] for r in scored]) for name in scored[0]}
            log({"updates": done, **summarize(metrics, "train", seeds),
                 **summarize(arrays, "fixed_mu_heldout", seeds)})
            write(directory/"status.json", {"status": "running", "updates": done, "utc": utc()})
        checkpoint("frozen-final", {"agent": agent, "history": history, "config": config})
        write(directory/"training-exit.json", {"exit_code": 0, "utc": utc(), "continuous_no_resume": True}, exclusive=True)
        summary = {"updates": budget["updates"], "fixed_encoder_actor_task_hash": frozen_hash,
                   "frozen_reference_sha256": reference_record["sha256"], "dataset_sha256": data_record["sha256"],
                   "all_seeds_retained": True}
    else:
        raise ValueError("unknown job kind")

    write(directory/"summary.json", summary, exclusive=True)
    for path in (metrics_file, directory/"summary.json"):
        enqueue_upload(directory, path, kind="results")
    write(directory/"status.json", {"status": "complete", "kind": job["kind"], "summary": summary, "utc": utc()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    study, admission = read(args.state/"study.json"), read(args.state/"admission.json")
    if not admission.get("production_admitted"): raise RuntimeError("new study is not admitted")
    verify_sources(ROOT, admission)
    job = next(j for j in study["jobs"] if j["name"] == args.job)
    directory = args.state/"jobs"/args.job
    ticket = read(directory/"ticket.json")
    if {k:v for k,v in ticket.items() if k != "gpu_uuid"} != job: raise RuntimeError("changed worker ticket")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ticket["gpu_uuid"]: raise RuntimeError("wrong GPU visibility")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "B200" not in devices[0].device_kind:
        raise RuntimeError("worker must see only its assigned B200")
    write(directory/"attempt.json", {"utc": utc(), "pid": os.getpid(), "training_retry_allowed": False}, exclusive=True)
    tracking = None
    try:
        import wandb
        if list(wandb.Api(timeout=60).runs(study["entity"]+"/"+job["project"], filters={"name":job["wandb_id"]}, per_page=1)):
            raise RuntimeError("scientific W&B ID already exists; refusing a duplicate")
        tracking = wandb.init(entity=study["entity"], project=job["project"], id=job["wandb_id"],
            resume="never", name=job["name"], job_type=job["kind"],
            group=f"{job['kind']}-{job['measure']}-{job['correction']}", config={"job":job})
        execute_job(args.state, job, directory, study, admission, tracking)
        tracking.finish()
    except Exception as error:
        successful_exit = (directory/"training-exit.json").exists() and read(directory/"training-exit.json").get("exit_code") == 0
        write(directory/"status.json", {"status": "finalization-failed" if successful_exit else "failed", "error":repr(error),
            "traceback":traceback.format_exc(), "utc":utc(), "training_retry_allowed":False})
        if tracking:
            try: tracking.finish(exit_code=1)
            except Exception: pass
        raise


if __name__ == "__main__": main()
