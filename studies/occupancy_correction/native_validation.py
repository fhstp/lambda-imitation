"""Bounded B200 end-to-end validation; never an admitted production learner."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .learner import create_learner, paper_settings
from .collection import start_rollout
from .sampling import prepare_episodes


ROOT = Path(__file__).resolve().parents[2]


def digest_tree(tree):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def validate_fixed_dataset(agent, environment):
    """Inspect actual stored transitions, not collector metadata alone."""
    reports = []
    for seed in range(agent.online_buffer.lengths.shape[0]):
        replay = jax.tree.map(lambda x: x[seed], agent.online_buffer)
        slots = np.flatnonzero(np.asarray(replay.lengths))
        lengths = np.asarray(replay.lengths)[slots]
        info = {name: np.asarray(value[slots]) for name, value in replay.info.items()}
        prepare_episodes(info, lengths, behaviour_policy_ids=["native-fixed-uniform-v1"] * len(slots))
        valid = np.arange(info["actions"].shape[1])[None, :] < lengths[:, None]
        if environment == "battleship":
            legal = info["observations"][..., 1:].astype(bool)
            assert np.all(legal[valid].sum(-1) > 0)
            assert np.take_along_axis(legal, info["actions"][..., None], -1)[..., 0][valid].all()
            expected = 1.0 / legal.sum(-1)[valid]
        else:
            expected = np.full(valid.sum(), 1 / 36)
        np.testing.assert_allclose(info["behaviour_probs"][valid], expected, rtol=2e-6)
        assert np.all(info["collector_version"][valid] == -1)
        reports.append({"retained_episodes": len(slots), "retained_transitions": int(lengths.sum()),
                        "logged_fixed_probabilities_verified": True})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("battleship", "minesweeper"), required=True)
    parser.add_argument("--measure", choices=("uniform", "discounted"), required=True)
    parser.add_argument("--correction", choices=("uncorrected", "is", "retrace"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("validation receipt already exists; preserve it")
    devices = jax.devices()
    assert len(devices) == 1 and devices[0].platform == "gpu" and "B200" in devices[0].device_kind
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible.startswith("GPU-") and "," not in visible
    settings = paper_settings(args.environment, measure=args.measure, correction=args.correction)
    seeds = [424200 + (100 if args.environment == "minesweeper" else 0) + i for i in range(5)]
    prefill_steps = 12800 if args.environment == "battleship" else 6016
    started = time.perf_counter()
    initial = [create_learner(settings, seed) for seed in seeds]
    fns = initial[0][2]
    agent = jax.tree.map(lambda *x: jnp.stack(x), *[x[0] for x in initial])
    rollout = jax.tree.map(lambda *x: jnp.stack(x), *[x[1] for x in initial])
    del initial
    prefill = jax.jit(jax.vmap(lambda a, r: fns.prefill(a, r, steps=prefill_steps)))
    agent, rollout = prefill(agent, rollout)
    jax.block_until_ready(agent.update_step)
    fixed_reports = validate_fixed_dataset(agent, args.environment)
    keys = jnp.stack([jax.random.fold_in(jax.random.key(s), 0x7A11) for s in seeds])
    train = jax.jit(jax.vmap(lambda a, r, k: fns.train(a, r, k, steps=64)))
    before = time.perf_counter()
    agent, rollout, keys, metrics = train(agent, rollout, keys)
    jax.block_until_ready(agent.update_step)
    first_seconds = time.perf_counter() - before
    assert np.all(np.asarray(agent.update_step) == 64)
    assert np.all(np.asarray(rollout.calls) == prefill_steps + 64)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((agent[:15], metrics)))
    before = time.perf_counter()
    agent, rollout, keys, metrics = train(agent, rollout, keys)
    jax.block_until_ready(agent.update_step)
    steady_seconds = time.perf_counter() - before
    assert np.all(np.asarray(agent.update_step) == 128)
    assert np.all(np.asarray(rollout.calls) == prefill_steps + 128)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((agent[:15], metrics)))
    evaluation = jax.jit(jax.vmap(lambda a, k: fns.evaluate(a, k, episodes=4, greedy=True)))(agent, keys)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(evaluation))
    assert np.all(np.asarray(evaluation["completed"]) == 1)

    # Separate fixed-mu data and fresh auxiliary fits. Only the reference's
    # encoder/actor/task networks are copied; no validation state is production.
    frozen_settings = replace(settings, frozen=True)
    frozen_initial = [create_learner(frozen_settings, s) for s in seeds]
    frozen_fns = frozen_initial[0][2]
    frozen = jax.tree.map(lambda *x: jnp.stack(x), *[x[0] for x in frozen_initial])
    from lambda_imitation.envs.battleship import Battleship
    from lambda_imitation.envs.minesweeper import MineSweeper
    frozen_env = Battleship() if args.environment == "battleship" else MineSweeper()
    frozen_shape = frozen_env.observation_space(frozen_env.default_params).shape
    frozen_steps = frozen_env.num_actions if args.environment == "battleship" else frozen_env.episode_length
    frozen_rollout = jax.tree.map(lambda *x: jnp.stack(x), *[
        start_rollout(frozen_env, jax.random.fold_in(jax.random.key(s), 0xDA71), frozen_shape, frozen_steps)
        for s in seeds])
    del frozen_initial
    frozen = frozen._replace(feature_extractor=agent.feature_extractor, actor=agent.actor, critic=agent.critic,
        feature_extractor_target=agent.feature_extractor, actor_target=agent.actor, critic_target=agent.critic)
    frozen_prefill = jax.jit(jax.vmap(lambda a, r: frozen_fns.prefill(a, r, steps=prefill_steps)))
    frozen, frozen_rollout = frozen_prefill(frozen, frozen_rollout)
    jax.block_until_ready(frozen.update_step)
    frozen_dataset_reports = validate_fixed_dataset(frozen, args.environment)
    assert digest_tree(frozen.online_buffer.info) != digest_tree(agent.online_buffer.info)
    before_hash = digest_tree(frozen[:3])
    fit = jax.jit(jax.vmap(lambda a, k: frozen_fns.fit(a, k, steps=64)))
    frozen, _, frozen_metrics = fit(frozen, keys)
    jax.block_until_ready(frozen.update_step)
    assert digest_tree(frozen[:3]) == before_hash
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((frozen[:15], frozen_metrics)))
    source_files = [*sorted((ROOT/"src").rglob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]
    receipt = {
        "passed": True, "validation_only_no_state_reuse": True,
        "environment": args.environment, "settings": asdict(settings), "seeds": seeds,
        "num_seeds": 5, "native_online_updates": 128, "native_frozen_updates": 64,
        "production_admitted": False, "cuda_visible_devices": visible,
        "device": str(devices[0]), "device_kind": devices[0].device_kind,
        "fixed_prefill_probability_checks": fixed_reports,
        "separate_frozen_dataset_checks": frozen_dataset_reports,
        "frozen_encoder_actor_task_hash_before_and_after": before_hash,
        "first_64_updates_seconds_with_compilation": first_seconds,
        "steady_64_updates_seconds": steady_seconds,
        "total_seconds": time.perf_counter() - started,
        "utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "flax", "optax", "numpy")},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "last_metrics": jax.tree.map(lambda x: np.asarray(x).tolist(), metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(receipt, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"passed":True,"environment":args.environment,"measure":args.measure,
        "correction":args.correction,"receipt":str(args.output),"total_seconds":receipt["total_seconds"],
        "steady_64_updates_seconds":steady_seconds,"frozen_networks_unchanged":True}),flush=True)


if __name__ == "__main__": main()
