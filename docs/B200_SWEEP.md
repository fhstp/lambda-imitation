# Battleship Bayesian search on8×B200

This is a **new,portable study**. Existing experiments/services/archives on the
development host are untouched. No inbound ports/server/dashboard required:
everything below is published through W&B outbound HTTPS. If outbound HTTPS
is unavailable,offline logging alone cannot run a W&B Bayesian controller.

## Budget recommendation

**First≈5 days:** broad Bayesian search,3 vmapped training seeds,**5M online
steps per seed**,plus12800 random-prefill steps (finish at natural boundary).
**Then:** narrower search/confirmation,5 or10 vmapped seeds and10M steps.
No performance pruning before1M,and no Hyperband in the default YAML:the old
experiments showed weak early returns are not a dependable failure signal.
Numerical/integrity failures fail closed rather than silently retrying.

Eight independent agents run one trial per GPU. Within a trial,training,prefill,
replay,optimizer moments,memory and RNGs are genuinely `jax.vmap`-batched across
seeds. Model construction and frozen memory analysis may iterate over seeds;
**training does not**. Each lane has an independent full replay and RNG.
Increasing seeds means a NEW trial/sweep,not adding seeds to an old checkpoint.

The default2048-trial cap is a safety ceiling,not a throughput prediction.
Benchmark a real full-suite trial before estimating the number possible in10
days. Dense memory fits can dominate. Pause the broad sweep around day5,let
active trials finish,and allocate remaining time to the narrower sweep. Keep a
margin before the lease expires;neither launcher assumes the lease renews.

## Setup on the B200 host

The branch is `feat/battleship-b200-sweep`. It must be pushed/transferred before
another host can fetch it;the coding agent does not push without approval.

```bash
git fetch origin
git switch feat/battleship-b200-sweep
PYTHON=python3.11 bash scripts/setup_b200.sh
source .venv/bin/activate
wandb login
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_PROJECT=battleship-b200-bayes
export BATTLESHIP_OUTPUT=/YOUR/LARGE/SCRATCH/battleship-bayes
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Python3.11/3.12 recommended. `requirements-b200.txt` fixes JAX0.6.2/Flax0.10.7/
Optax0.2.5 with CUDA12 wheels. **Blackwell needs a CUDA12.8+ compatible runtime
and driver**;inspect `nvidia-smi`,installed CUDA wheel packages and actual GPU
validation. Do not assume the old development host's CUDA/JAX setup works on
B200. This target profile requires real-device validation;local CPU tests are
NOT a B200 certificate. Installation writes `b200-environment.txt` for provenance.
Gymnax is pinned to0.0.9 (the supported five-value step API),not1.x's changed
step signature. Setup uses an isolated venv,never system-site-packages:mixing
old CUDA plugins with a newer jaxlib can crash even during CPU backend discovery.
Use the site's approved container if its driver/runtime requires one;preserve
and log the actual versions,and rerun ALL native gates after changing versions.

## Mandatory preflight,then launch

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -q
CUDA_VISIBLE_DEVICES=0 python examples/battleship_sweep.py --validate-only --diagnostics-smoke --output "$BATTLESHIP_OUTPUT/preflight"
cp sweeps/memory-policy.example.json memory-policy.json
wandb sweep --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" sweeps/battleship_bayes.yaml
# Copy the entity/project/sweep_id printed by the previous command:
bash scripts/battleship_agents.sh ENTITY/PROJECT/SWEEP_ID 8 256
```

Use tmux or the site's batch scheduler;no forwarded ports. Eight trial processes
each see one GPU via CUDA_VISIBLE_DEVICES. The diagnostic smoke uses ONE seed,
300 episodes per mode/200 fit updates/representative horizons and publishes
W&B tables/plots/images/artifacts. Production uses900 per mode/2000 fit updates.
The native training gate uses the configured full network/replay/batch/seed
count and256 updates,exact same-kernel repeatability,lane-isolation and full
host→device checkpoint-resume tests. **Every trial revalidates its actual
hyperparameters/shapes** and discards validation states before fresh training.
No scalar fallback,precision downgrade or tolerance relaxation on a failure.
Local CPU verification and its limits are recorded in [B200_VALIDATION.md](B200_VALIDATION.md).

Earlier legacy categorical vmap repeatability gates FAILED and remain archived.
This new runner replaces colliding categorical-projection scatter-adds with
the mathematically equivalent dense triangular interpolation/reduction. That
removes one potential atomic source of nondeterminism;it is **not proof** no
other nondeterminism remains. Projection law/mass/mean tests and the actual
repeat/resume/isolation gate are required. Native trajectories can differ from
the old scalar/scatter implementation;this is not bitwise legacy equivalence,
nor an assertion that the old failures were conclusively diagnosed.

## Search space and objective

`sweeps/battleship_bayes.yaml` is ready for `wandb sweep` (method:`bayes`).

| Parameter | Distribution | Range / values | Reason |
|---|---|---|---|
| λ-discrepancy coefficient β | log_uniform_values | .03…100 | Includes weak through strong pressure around observed β10 benefit |
| Memory/projection LR | log_uniform_values | 1e−6…3e−4 | Coordinate drift vs learning speed;current reference1e−4 |
| Actor LR | log_uniform_values | 3e−5…1e−3 | Actor–critic mismatch was substantial |
| Main+aux critic LR | log_uniform_values | 3e−5…1e−3 | Critic tracking/calibration;aux tied to main |
| Final α | log_uniform_values | .003….3 | Exploration/coverage versus exploitation |
| α decay steps | categorical | 100k,250k,500k,1M | Discrete meaningful schedule scales |
| Target EMA τ | log_uniform_values | .001….02 | Target lag/tracking |
| GRU width | categorical | 128,256,512 | Memory capacity |

Scale parameters vary by orders of magnitude,so use log-uniform **values**,not
uniform or `log_uniform` with incorrectly interpreted endpoints. No arbitrary
continuous parameter here warrants uniform sampling. Width/schedule are discrete.
β0 cannot be sampled from a logarithmic range;`sweeps/beta0-control.json` is an
explicit matched-reference config,not a claim that the controller samples zero:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/battleship_sweep.py --config sweeps/beta0-control.json
```

Architecture fixed:GRU+prev-action projection128,residual actor128/critic256,
two expansion4 blocks,fixed pre/post RMS,criticOFN,all6 critics51 atoms[-1.5,1.5].
Full64-step episodes,masked early-terminal padding,batch128,replay100k,gamma.99,
lambda .05/.85,reward-only projected main law/scalar Retrace two-hot auxiliaries,
detached-head scalar min-mean discrepancy,actor-to-FE stopped. No action masking.
Monitor clipping;bounded categorical Q does NOT imply calibration.

One W&B run = one configuration +ALL seeds. Bayesian objective is the mean over
training seeds of the last5 fixed-schedule stochastic-evaluation returns at
**the final equal5M budget**,not the best seed/checkpoint or probe accuracy.
It is logged ONLY on successful completion;intermediate runs do not supply
incomparable partial scores. At5M those checkpoints are3M/3.5M/4M/4.5M/5M.
Memory-schedule changes do not change this evaluation schedule/objective.
Finalists use a different sweep/budget;do not compare its objective directly
with5M search scores as though budgets matched. Seed std is descriptive,not a
confidence interval or winner-generalization guarantee.

## Memory metrics,full curves and visuals in W&B

Default cadence:10k,20k,…100k;200k,…1M;1.5M,…5M;6M,…10M.
That is27 observations in5M or32 in10M. Each due observation runs:

- Frozen stochastic/greedy/uniform histories,900 episodes PER mode;
  episode-disjoint600/100/200 train/validation/test,original fixed probes.
- Full-history memory linear/MLP128 decoding shots/unique shots/hits and fired/
  revealed-hit masks. MAE/RMSE/R²,balanced accuracy,per-class recall,AUROC,
  IoU/exact-board metrics,age/last-fire-lag slices in JSON artifacts.
- Recent-input,clock+recent-input,train-only prevalence and same-mode/age
  shuffled-memory controls. No test-based probe selection or threshold tuning.
- Independent lastN-window AND exact-lag probes N/k1…64 from the SAME post64-
  shot carry;earlier-terminal episodes excluded with counts. Linear/MLP memory,
  recent input,prevalence and within-mode shuffled controls. Fixed2000-update
  fits,per-episode AUROC and pointwise200-episode bootstrap intervals/support.
- W&B native line-series plots for every window/lag×fired/hit×policy mode,
  complete per-seed tables with intervals/support,selected-horizon scalar time
  series,per-seed values and mean/std/count. Missing/undefined metrics are
  unavailable,not zero-filled;available-seed means disclose contributing count.
- Quality-independent heldout visual examples:four8×8 maps (observed history,
  score,original-validation-threshold binary,error) inside W&B image tables.
  Cell-level tables/JSON enable richer later analysis without a local server.
- Policy repeat mass/shot repeats/unique coverage/entropy/returns,current-policy
  sampled MC Q grounding,main/aux target clipping/edge mass,Q/OFN norms,replay
  actor-vs-main-Q grounding and same-batch FE main/aux/discrepancy gradient norms,
  true combined-vector norm and cosine. Diagnostic float32 matmul stays local
  to gradient/grounding audits;training precision remains native.

Look for `memory/curves/...`, `memory/maps/SEED`, `memory/horizon_table/SEED/...`,
`memory/full_history/...`, `memory/horizon/...`, `memory/grounding/...`, and
`seed/SEED/...`. Complete raw measurement JSONs are versioned W&B artifacts.
Tables are partitioned to stay under W&B's10k-row history limit.

Decodability ≠ policy use/causal return benefit. These targets are OBSERVED
fired/revealed-hit history,never unseen ships. Revealed-hit negatives are NOT
confirmed water. Scores are class-weighted probe scores,not calibrated ship
probabilities. Uniform histories match;the mixed-policy probe-training mixture
still changes with checkpoint/hyperparameters. Exact-lag hits can be very sparse;
no automatic hard horizon,simultaneous-CI or oracle-Q claim. Q-return comparisons
on greedy/random rollouts change the future policy;only stochastic-current
rollouts are on-policy MC grounding,still not exact representation-conditional Q.

## Reduce observation frequency without stopping agents

The trial polls `runtime_policy` after each10k steps,not during a diagnostic.
It is a **separate observation policy**,not a hyperparameter override. Changes
are timestamped locally and recorded as W&B tables/artifact metadata. For fewer
full-suite checkpoints,atomically replace `memory-policy.json`:

```json
{
  "memory_schedule": "milestones",
  "memory_suite": "full",
  "memory_milestones": [100000, 1000000, 3000000, 5000000, 10000000]
}
```

Write a temporary file,then `mv` it into place;partial/invalid JSON fails closed.
`memory_suite:"tiered"` instead keeps the full-history probes and only window/
lag1/5/10/16/32/64. **Default is full at every requested observation.** Final
memory analysis always runs. Learning settings and fixed return evaluation
schedule never change through this file. Different observation policies affect
cost,not the training RNG,which is hash-certified unchanged by diagnostics.

## Day5 narrowing and5/10 seeds

```bash
wandb sweep --pause ENTITY/PROJECT/BROAD_SWEEP_ID
# Allow active trials to finish. Generate,inspect,then register a NEW sweep:
python scripts/narrow_battleship_sweep.py ENTITY/PROJECT/BROAD_SWEEP_ID --top 12 --n-seeds 5 --output sweeps/battleship_narrow.yaml
wandb sweep --entity "$WANDB_ENTITY" --project battleship-b200-finalists sweeps/battleship_narrow.yaml
bash scripts/battleship_agents.sh ENTITY/battleship-b200-finalists/NEW_SWEEP_ID 8 16
```

The helper uses only completed equal5M/3-seed trials with finite final scores.
It narrows log ranges to selected min/max with×1.5 padding inside original
bounds,and categorical values to selected ones. It writes selection provenance,
does NOT launch anything or overwrite an existing YAML. Review ranges rather
than treating the generated prior as proof of optimality. `--n-seeds 10` works
unchanged;fresh seed base1000 reduces training-seed reuse after selection.

## Storage,resume and limits

`artifact_policy:metrics` retains complete measurement JSONs,all curve values,
selected heldout reconstructions and certificates,uploads them to W&B,and
discards ONLY this NEW trial's temporary fit intermediates. No existing archives
are deleted. Full raw per-episode memories/probe parameters/every prediction
are NOT retained by default;unretained states cannot be re-audited later.
`artifact_policy:all` additionally keeps/uploads all raw datasets/models/scores;
this is extremely expensive at27/32 observations×3–10 seeds. Benchmark storage.
Do not confuse complete *metric curves* with full raw-data retention.

Rolling local full-state checkpoint at each evaluation/diagnostic,final full
state as a W&B artifact,including ALL seed model/replay/optimizer/rollout/RNG.
Local files for completed runs remain;no automatic artifact deletion. Expect
roughly1–several GiB per3-seed metrics-only trial depending on table/JSON/media
encoding,and much more forraw retention. Mount large scratch and monitor W&B
quota;default local free-space reserve10GiB fails safely. A lease interruption
can lose work after the latest checkpoint but never silently rerun prefill:

```bash
python examples/battleship_sweep.py --resume /SCRATCH/RUN_ID/latest.pkl --output /SCRATCH
```

Resume requires the SAME source manifest,full configuration,seed count and
W&B run ID and package versions. It rebuilds/revalidates kernels then restores exact saved state;
it does not claim cross-device/kernel-version bitwise trajectories. Don't run
two resumes for one trial simultaneously. No automatic failure retries.
An initial post-prefill step0 checkpoint is saved too. If interruption leaves
a later diagnostic directory beyond the last checkpoint,inspect it and move it
to a **distinct preserved pathname** before explicitly resuming. The runner
refuses to overwrite an existing complete/partial diagnostic attempt. It never
silently skips/reuses a partial measurement or deletes old evidence.

`--cpu-smoke` is a tiny3-lane CPU safety test,not a production certificate;
`--cpu-smoke --diagnostics-smoke` also checks reduced-budget probe/W&B plumbing.
Add `--full-memory-smoke` to test all128 window/lag tasks at the reduced fit budget.
Never use smoke state/data to initialize a sweep. No B200 speedup/safety claim
is made until the target host's gates pass.
