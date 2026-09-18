# Selected AC+lambda implementation

This branch contains the computation active in **F5-lambda-off**, continued by
**A2-lambda-off**. It is a focused extraction for reviewing changes against local
`main`, not a merge of the experimental ladder.

## Provenance

| Identity | Value |
|---|---|
| Main/base | `6af9158bc6b59504fac0b109574271f537bf4018` |
| Complete study snapshot | `805d48169138ec82f73323cb850d2706643daab6` |
| Active 124-file source manifest SHA-256 | `4fd418701fc13c0e5281beeec12275d185f9ddfb7443aa5fe608227b89154524` |
| Active continuation resolved config SHA-256 | `52873d22c0f2b60e0c9f066e3ffc2fc75262c62dc3e10cd355d31d196a0e9547` |
| Original 500k F5 config SHA-256 | `95ac739a28cdde45ad44b5af0d8e9280eeed0ff0852c560d25a8120a256da50d` |
| Dependency lock SHA-256 | `9e45fa9fb3ca8b060cb6638c92402b85f9570b96ad344c5284cebdc2af4caef4` |
| Upstream environment commit | `brownirl/lambda_discrepancy@e0c027df237915c7da28c13e9426554f977486bd` |

The dependency lock is retained intact to identify the validated runtime. It
includes dependencies from the study; this extraction does not import its PPO,
GVD, sweep, probe, remote-execution or recovery-controller implementations.

## Review map

| File | Changes that matter |
|---|---|
| `src/lambda_imitation/ac_lambda.py` | Selected initialization, recurrent forwards, five objectives, separate Adam rates, EMA, collection and evaluation |
| `src/lambda_imitation/episode_replay.py` | Complete-natural-episode FIFO over a transition ring; unchanged active implementation |
| `src/lambda_imitation/ac_lambda_train.py` | One fresh-training/resume entrypoint, resolved config, immutable checkpoints and runtime hashes |
| `src/lambda_imitation/iqlearn.py` | Only active `Head` list-container compatibility with locked Flax |
| `src/lambda_imitation/utils.py` | Only removal of initialization RNGs from recurrent optimizer state |
| `vendor/lambda_discrepancy/` | Unmodified Battleship environment and Apache license; no upstream trainer |
| `tests/test_ac_lambda.py` | Scalar recursion, gradient routes, replay, resume, CLI and actual active-producer parity |

Source-function mapping within the study snapshot:

- `battleship_controls.initialize_model`, `battleship_five.initial_state`,
  `battleship_explore.normalized_head/adapt` and `battleship_fe_addons.adapt`:
  canonical seven-head initialization and deterministic critic LayerNorm.
- `pfe.PilotModel`, `pfe.policy/selected/valid_mean/collect/evaluate`:
  recurrent and action-mask semantics, stream keys, terminal resets, evaluation.
- `pfe.objectives` supplies standard TD and discrepancy;
  `battleship_combined.objectives` replaces actor and lambda TD;
  `quick_controls.q_conditioned_targets` supplies the active lambda recursion.
- `battleship_fe_addons.update/training_step/kernels`: one gradient snapshot,
  FE/head Adam rates, EMA and insert-one/update-one cadence.
- `pfe_replay.py`: retained verbatim as `episode_replay.py`.

No abandoned producer functions are extraction sources. They are preserved only
in the complete study snapshot. The generic existing project API remains usable;
the extracted trainer avoids adding experimental flags to it.

## Exact selected computation

- Native 5×5 Battleship, ships `(3,2)`, 25 legal-masked actions, sparse −1/+25
  rewards. Only previous hit plus previous executed-action one-hot enters memory.
- Existing projection 1024→512 with hit skip, GRU512; actor hidden512, standard
  and two lambda critics each independent twins with hidden512. Critic hidden
  Linear→LayerNorm→ReLU, epsilon `1e-6`, fast variance, float32, scale1/bias0.
  Actor/FE have no LayerNorm. Seven named head keys retain their original order.
- Shared FE Adam `1e-5`, all heads `1e-4`; beta1 `.9`, beta2 `.999`, epsilon
  `1e-8`, eps_root0, no clipping/annealing. Five group applications per update,
  all from one pre-update snapshot, followed by all EMA updates at tau `.001`.
- Gamma `.99`, fixed actor alpha `.1`, **no entropy in standard bootstrap**.
  Entire actor Q teacher is stopped; actor/entropy still train shared FE.
- Executed-action standard half-sum twin MSE; lambda min-twin half-MSE; executed
  min-twin Huber discrepancy, weight1, with lambda-head parameters stopped.
  All losses use one global genuine-row mean, including terminal rows.
- Lambda values `.1/.95`; real `pi_EMA/mu`, fake-on-policy false. For each lambda:

  ```text
  V_t = sum_a pi_EMA(a|H_t) min(Q1_EMA,Q2_EMA)(H_t,a)
  c_t = lambda * min(1.1, pi_EMA(a_t|H_t)/mu_t)
  G_t = r_t + .99*(1-d_t) * [V_(t+1) + c_(t+1)*(G_(t+1)-Qbar_(t+1)(a_(t+1)))]
  ```

  Targets are entirely stopped. There is no current-rho multiplier; terminal
  target is actual terminal reward. No GVD heads, draws, optimizer state or losses.
- Replay capacity200k **transitions**; 128 complete natural episodes sampled
  uniformly with replacement, padded to25. Independent online/EMA full BPTT
  from true-start zero carry/action. Unfinished episodes stay unsampleable;
  ring overwrite evicts incomplete retained prefixes. Burn-in0/target-tail0.
- Natural uniform-legal prefill16640, then collect-one/update-one. Carry and
  previous action reset together only on actual terminal, never at a block,
  reporting or prefill boundary. Root2029/split10 seed indices; four-way stream
  split per collection. Default seed0; index9 is the validation fixture.
- Common500-episode stochastic evaluation uses fixed-bank domain2 and folds
  `0xE24/0xA24`, then per-episode fold-in keys. Evaluation does not consume
  training RNG. Return equals26−episode length.

`resolved_config()` serializes all settings and fixed algorithm choices into
every run. CLI accepts only seed, total horizon, output and optional checkpoint;
unknown flags fail. A fresh500k run has516640 training interactions; a fresh2M
run has2016640. Continuing a full500k checkpoint to2M adds1.5M updates without
prefill or initial re-evaluation. Evaluations are counted separately.

The active producer computes separate recurrent forwards for standard/LD and
actor/lambda TD. This structure is intentionally retained: merging the forwards
failed the strict full-width float32 gradient check. Overwritten inactive losses
are removed while preserving active gradient accumulation order.

## Run from this checkout

Use Python3.12 and the retained lock in a separate environment:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-study.lock

PYTHONPATH=src JAX_ENABLE_X64=0 .venv/bin/python -m lambda_imitation.ac_lambda_train \
  --seed 0 --total 500000 --output /path/to/new-500k-run

PYTHONPATH=src JAX_ENABLE_X64=0 .venv/bin/python -m lambda_imitation.ac_lambda_train \
  --seed 0 --total 2000000 --resume /path/to/new-500k-run/step-00500000 \
  --output /path/to/new-continuation
```

For a fresh2M run, omit `--resume`. Leave matmul precision at default and select
an available backend/device before launch. These examples do not authorize
additional study runs or sharing an occupied GPU.

Milestones are0/100k/250k/500k and, for2M,750k/1M/1.5M/2M. Nonfinal payloads are
evaluation-only FE+actor; final and explicit Ctrl-C checkpoints contain full
parameters/targets/Adam/replay/pending/env/carry/action/RNG/counters. Ctrl-C retains
the last returned complete block. Output directories must be absent; checkpoint
publication is atomic and collision-rejecting.

The small standalone checkpoint format is **not a loader for historical study
artifacts**. The running study's 500k→2M continuation remains owned by its original
controller. Absolute artifact wrappers, hardware admission, remote transport,
historical recovery and multi-job queues are deliberately outside this branch.

## Validation and limits

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m pytest tests/test_ac_lambda.py -q -p no:cacheprovider
```

Two additional oracle checks require `AC_LAMBDA_STUDY` pointing to the approved
frozen study workspace. They call its actual active F5 functions, never copied
oracle implementations. Shared network definitions are checked by AST and
resolved numerical fields against the active continuation fixture.

Validation includes exact initialization, targets, five losses/metrics, all
gradients, two Adam/EMA updates, real collection/replay and terminal carry/action
resets, evaluation keys/rows, stopped gradient routes, replay eviction/counter
rollover, exact checkpoint roundtrip and resumed updates. Full-width512/T25/B2
CPU parity uses `atol=rtol=1e-6`; integer fields and initialization are exact.
Small-width8 tests additionally exercise production B128/T25 training steps.
CLI horizon routing uses explicitly mocked neural boundaries; actual kernels
are tested separately. Existing carry/projection/wrapper regressions also run.

Recorded CPU results (2026-09-18, locked runtime, affinity2–5, GPUs hidden):
full-width parity **1 passed /23.46s**; remaining extraction tests plus existing
`TestCarryLayout`, `TestProjection`, `TestWrapperEquivalence` **43 passed /67.16s**.
The full-width node was deselected in the latter run because its unchanged final
core had already passed. Original failed merged-forward comparison is retained
in the task's validation evidence, not relabeled as passing.

This establishes deterministic short-fixture equivalence on CPU, not long-run
bit identity across different compiled graphs or hardware. No new training run,
GPU parity run or scientific performance claim is part of this extraction.
