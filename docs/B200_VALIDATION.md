# Local validation (15 October2026)

**Status: CPU-verified; B200/CUDA launch NOT yet certified.** No production
sweep was launched and no legacy experiment/service/plan was modified.

## Clean target-stack CPU profile

Isolated Python3.12 venv,without system-site-packages:

| Package | Version |
|---|---|
| jax / jaxlib | 0.6.2 / 0.6.2 (CPU) |
| flax | 0.10.7 |
| optax | 0.2.5 |
| numpy | 1.26.4 |
| gymnax | 0.0.9 |
| popjym | 0.0.1,pinned Git revision from pyproject.toml |
| cloudpickle | 3.1.2 |
| wandb | 0.30.0 |

- **123 repository tests passed**,including configuration restrictions,exact
  cadence,projection mass/mean/legacy mathematical reference,actual seed-vmap
  wrappers,toy orchestration/final scoring/resume,narrowing,undefined metrics,
  independent average-rank AUROC reference and repeat-miss hit-target semantics.
  One existing pytest fixture-scope deprecation warning remains.
- Tiny native3-lane training gate passed same-kernel **byte-exact** repeatability,
  lane isolation,host serialization→device restore/resume and finite-state
  checks. Source manifest independently checked against the final runner.
  Tiny network/replay/batch and2 updates:NOT the full-shape production gate.
- Reduced-fit frozen memory audit completed **all128 tasks** (64 window+64 lag),
  **384 independent horizon fits**,plus the six full-history fits and controls.
  One seed,300 episodes per policy mode,200 updates per fit;200-bootstrap
  intervals. Frozen state digest unchanged. No smoke state/data is reused.
- Offline W&B publication completed:16 native curve plots,22 partitioned
  history tables (maximum7680 rows),complete diagnostic JSON artifact and120
  four-map reconstruction rows. Typed artifact serialization independently
  verified120 image references and116 deduplicated,existing PNG payloads.
  History-table JSON uses image placeholders;actual images are typed artifact
  media,not necessarily loose PNGs under the offline run's `files/media`.
- Both shell launch/setup scripts pass `bash -n`;`git diff --check` passes.

Local evidence (development machine only,not needed on the server):

- `/tmp/opencode/b200-final-byte-gate/cpu-smoke/validation.json`
- `/tmp/opencode/b200-profile-full-validation-attempt1/cpu-smoke/diagnostic-proof.json`
- `/tmp/opencode/b200-profile-full-validation-attempt1/cpu-smoke/diagnostics/step0/seed42/result.json`
- `/tmp/opencode/b200-wandb-full-clean/wandb/offline-run-20261015_114308-3azn471t/`

## Failures found and preserved

Earlier attempts were kept rather than overwritten:

1. Calling an old `config_for` reached host-specific plan/package guards.
   Replaced with standalone configuration construction;no old guard was bypassed
   or weakened. Regression test forbids those calls.
2. Mixing the newer jaxlib with system-installed older CUDA plugins segfaulted
   backend discovery even with CPU selected. Rebuilt an isolated environment;
   the old environment was not edited. Remote setup uses the same isolation.
3. Fresh package metadata rejected the pinned direct Git dependency until
   Hatch's explicit `allow-direct-references` option was added.
4. Unconstrained Gymnax1.0 changed its step return signature and broke older
   general environment tests. Pinned0.0.9/supported0.x API;the full suite passes.
5. The inherited buffer-contiguity assertion ignored circular wraparound.
   Applied the existing development-tree correction (differences modulo ring
   size);no replay implementation or learning behavior changed.

These are separate from the earlier archived **legacy categorical GPU-vmap
repeatability failures**,which remain failures. The new dense projection is an
explicit new-study implementation,not a retroactive passing certificate.

## Mandatory on-server checks

Follow [B200_SWEEP.md](B200_SWEEP.md). The real CUDA/B200 kernel,full production
shapes/configuration and3/5/10-seed count must pass the native gate. Each trial
does this before training. Run the diagnostic smoke with W&B **online** to
confirm actual outbound cloud publication and rendering on the leased host.
Benchmark production-budget diagnostics before estimating trial throughput,
storage or whether10 days can cover a particular number of configurations.
No GPU throughput,online dashboard-rendering or cross-device bitwise guarantee
is inferred from local offline/CPU success.
