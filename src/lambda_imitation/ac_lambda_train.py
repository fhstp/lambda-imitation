"""Fresh selected AC+lambda training and full-state 500k -> 2M continuation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from . import ac_lambda as ac


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def publish(path, files):
    """Same-filesystem atomic directory publication; never replace a checkpoint."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as temporary:
        root = Path(temporary)
        for name, raw in files.items():
            with (root / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        # Single output owner enforced by exclusive output mkdir in main.
        root.rename(path)


def save_checkpoint(path, state, config, *, full, returns=None):
    value = state if full else {k: state.learner.online[k] for k in ("actor_fe", "actor")}
    leaves = [np.asarray(x) for x in jax.tree.leaves(value)]
    if not bool(state.healthy) or not all(np.isfinite(x).all() for x in leaves):
        raise ValueError("unhealthy/nonfinite checkpoint")
    payload = serialization.msgpack_serialize({"leaves": leaves})
    meta = dict(format=1, config=config, config_sha256=digest(canonical(config)),
        full=full, updates=int(state.interactions), prefill=int(state.prefill),
        payload_sha256=digest(payload),
        leaves=[dict(shape=x.shape, dtype=str(x.dtype)) for x in leaves])
    files = {"state.msgpack": payload}
    if returns is not None:
        rows = {k: np.asarray(v).tolist() for k, v in returns.items()}
        files["returns.json"] = canonical(rows)
        meta["returns_sha256"] = digest(files["returns.json"])
    files["manifest.json"] = canonical(meta)
    publish(path, files)


def restore_checkpoint(path, template, config):
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_bytes())
    old = meta["config"]
    if (meta["format"] != 1 or not meta["full"] or
            digest(canonical(old)) != meta["config_sha256"] or
            canonical(old) != canonical(ac.resolved_config(old["seed"], old["total"])) or
            old["seed"] != config["seed"] or old["settings"] != json.loads(canonical(config))["settings"] or
            meta["updates"] > config["total"]):
        raise ValueError("incompatible checkpoint/config or evaluation-only payload")
    raw = (path / "state.msgpack").read_bytes()
    if digest(raw) != meta["payload_sha256"]:
        raise ValueError("checkpoint payload hash mismatch")
    actual = serialization.msgpack_restore(raw)["leaves"]
    expected, structure = jax.tree.flatten(template)
    if len(actual) != len(expected) or len(meta["leaves"]) != len(expected):
        raise ValueError("checkpoint leaf count mismatch")
    for x, y, info in zip(actual, expected, meta["leaves"], strict=True):
        if (x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(x).all() or
                info != dict(shape=list(x.shape), dtype=str(x.dtype))):
            raise ValueError("checkpoint leaf shape/dtype/finiteness mismatch")
    state = structure.unflatten([jnp.asarray(x) for x in actual])
    if (int(state.interactions) != meta["updates"] or int(state.prefill) != meta["prefill"] or
            int(state.learner.updates) != meta["updates"] or not bool(state.healthy) or
            int(state.replay.inserted) != meta["updates"] + meta["prefill"] or
            any(int(v[0].count) != meta["updates"] for v in state.learner.optimizers.values())):
        raise ValueError("checkpoint counters/health mismatch")
    return state


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new, absent output directory")
    p.add_argument("--seed", type=int, default=0, choices=range(10))
    p.add_argument("--total", type=int, default=500000, choices=(500000, 2000000))
    p.add_argument("--resume", type=Path, help="full checkpoint produced by this entrypoint")
    return p


def main():
    args = parser().parse_args()
    if jax.config.jax_enable_x64 or jax.config.jax_default_matmul_precision is not None:
        raise ValueError("selected computation requires x64 disabled and default matmul precision")
    config = ac.resolved_config(args.seed, args.total)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "resolved-config.json").write_bytes(canonical(config))
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__), Path(ac.__file__), Path(ac.replay.__file__),
                    root / "src/lambda_imitation/utils.py", root / "src/lambda_imitation/iqlearn.py",
                    root / "vendor/lambda_discrepancy/lamb/envs/battleship.py", root / "requirements-study.lock"]
    runtime = dict(config_sha256=digest(canonical(config)), source={str(p.relative_to(root)):digest(p.read_bytes()) for p in source_paths},
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64), precision=jax.config.jax_default_matmul_precision,
        packages={k:importlib.metadata.version(k) for k in ("jax","jaxlib","flax","optax","numpy","gymnax")},
        resume=None if args.resume is None else dict(path=str(args.resume.resolve()),
            manifest_sha256=digest((args.resume/"manifest.json").read_bytes())))
    (args.output / "runtime.json").write_bytes(canonical(runtime))
    model, learner = ac.initialize_model(args.seed)
    env, state = ac.initial_state(model, learner, args.seed)
    prefill, train = ac.kernels(model, env)
    resets, actions = ac.evaluation_keys(args.seed)
    evaluate = jax.jit(lambda online: ac.evaluate(model, env, online, resets, actions))

    def measurement(state):
        rows = jax.block_until_ready(evaluate(state.learner.online))
        if (not np.asarray(rows["completed"]).all() or not np.asarray(rows["legal"]).all() or
                not np.array_equal(rows["returns"], 26 - rows["lengths"])):
            raise ValueError("invalid evaluation")
        state = state._replace(evaluation_interactions=state.evaluation_interactions + rows["lengths"].sum())
        save_checkpoint(args.output / f"step-{int(state.interactions):08d}", state, config,
                        full=int(state.interactions) == args.total, returns=rows)
        print(json.dumps(dict(updates=int(state.interactions), mean_return=int(rows["returns"].sum())/500)), flush=True)
        return state

    if args.resume is not None:
        state = restore_checkpoint(args.resume, state, config)
        if int(state.prefill) != 16640:
            raise ValueError("resume requires completed natural prefill")
    else:
        state = measurement(state)
        state = jax.block_until_ready(prefill(state, 16640))
    try:
        while int(state.interactions) < args.total:
            count = min(10000, args.total - int(state.interactions))
            state, metrics = jax.block_until_ready(train(state, count))
            if not bool(state.healthy):
                raise ValueError("collection/replay/update health failed")
            print(json.dumps(dict(updates=int(state.interactions), training_interactions=int(state.interactions+state.prefill),
                                  metrics={k:float(v) for k,v in metrics.items()})), flush=True)
            if int(state.interactions) in config["checkpoints"]:
                state = measurement(state)
    except KeyboardInterrupt:
        save_checkpoint(args.output / "interrupted", state, config, full=True)
        raise


if __name__ == "__main__":
    main()
