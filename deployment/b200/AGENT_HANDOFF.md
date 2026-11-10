# Scientific and operational handoff

## Objective and authorization

Paper: *Retracing the λ-Discrepancy: Memory Learning from Off-Policy Data*.
Reviewer: “Can you extend Figure 6's analysis to controlled memory
representations or history-window lengths on the learning benchmarks? In
particular the non-monotonicity of the discrepancy-measure is concerning.”

The user approved Battleship AC+Retrace only, with histories
**0,1,2,5,10,15,20,25**, ten seeds each, keeping the paper settings otherwise.
The original agent selected and explained the controlled rolling-window
definition below. The user later approved a portable deployment on another
eight-B200 server behind a custom VPN. They may not have permission to use
OpenCode there. The deployment must work without an agent or systemd.

## Fixed study definition (do not change)

- Base: `publication-cleanup`, commit `07381e766181521b3461b87b274970f00d8b46d1`.
  This branch is `rebuttal/history-windows-2026-11-09`; do not use a separate
  research checkout on `test/max_D` as a base.
- H counts previous network inputs in addition to the current input. Each
  input is latest shot outcome + previous action. The same 512-unit GRU is
  run from zero over each rolling window, with episode-safe invalid padding.
  Only raw inputs are retained; no learned carry from before the window.
  H=0 is one-step GRU, not the paper's separately tuned identity baseline.
- Identical embedding/GRU/head parameter count and initialization semantics
  across H. Actor gradients stop at features even at H=0.
- Paper legal-action mask is unchanged. It encodes all previous shot locations:
  this caps **representation history**, not all information available to policy.
- MSE auxiliary twin regression, Retrace, λ=(0.05,0.85), β=0.01; learning rates,
  γ=0.99, α=0.1, τ=0.005 and all other flags are in the sealed protocol.
- Common extra context-only replay prefix 26 at every H; original burn-in 10,
  learning sequence 40 and look-ahead 50 unchanged. Replay allocation becomes
  126 instead of historical 100. Actor scores 90 positions; task/discrepancy
  89; auxiliary regression 40. Prefill stays explicitly **12,800**, not 128×126.
- Fifty rounds ×10,000 online interactions per seed, 50 evaluation episodes.
  Report the same 48 checkpoints through **492,800** total interactions in the
  paper's ≤500,000 window. Final training ends at 512,800 including prefill;
  do not use the extra endpoint as the primary result or select a peak.
- Primary discrepancy is per-seed RMS of current-policy-action-weighted online
  twin-min head disagreement on independent sampled-policy evaluation episodes.
  Return is greedy evaluation return. Replay RMS, greedy-trajectory RMS,
  auxiliary twin gaps and regression losses are secondary fitting diagnostics.
- Mean and pointwise 95% Student-t CIs across independent training seeds.
  Adjacent-H contrasts pair nominal seed labels, exploratory without multiple
  testing adjustment. No pooling of historical or repeated runs as extra seeds.
- This is fitted critic disagreement under separately learned policies and
  visitation, NOT an exact fixed-point norm or fixed-policy monotonicity test.
  Larger H does not guarantee monotone discrepancy/return; a zero score is
  not a sufficiency certificate. Do not retune to manufacture monotonicity.

## Work allocation and current-state source of truth

The original RTX Pro 6000 host runs first blocks: five seeds starting
`7451234536`. This B200 server exclusively owns **all eight second blocks**:
five seeds starting `7451234542`. Exact W&B IDs/sessions/flags and the issued
lease are in `assignment.json`. These jobs have already been excluded from
the original dispatcher. Never launch first blocks here or run this same
assignment on another host. Do not edit assignment/protocol/run IDs.

At reservation time (2026-11-10), six first blocks were training locally,
H=0 first block was completed, and H=1 first block was still queued there.
This is a historical snapshot, NOT a live claim. All continuous replay-
correction reruns had finished. Fresh PPO+λ training/rollouts were separately
authorized earlier but are NOT part of this B200 assignment; do not launch
them or other benchmarks from this handoff.

Read `run.sh status`, state `settings.json`, `status.json`, per-job
`ticket.json`, `production-started.json`, `training-exit.json` and logs, GPU
processes and W&B for actual current state. Do not infer that a ticket means
successful policy initialization or that a native proof is a completed run.

## Source seals and validation

`reviewer_response/admission.json` and `source_manifest.json` bind the exact
learning code, runner, protocol and original validation receipts. Original
evidence stays in place. Portable native validation redirects only its output
root to persistent state, never overwriting `reviewer_response/native-h*.json`.
`deployment/b200/bundle-manifest.json` separately binds deployment code and
instructions. Do not edit a hash-bound file while workers are active.

The original study passed CPU regression tests and all eight native full-shape
checks on RTX Pro 6000, plus a real checkpoint upload/download hash canary.
Packaging adds CPU tests for assignment, isolation, lifecycle and import guards.
**B200 access is not available to the original agent: no successful remote
install/preflight or B200 speed claim exists until this server records one.**
The queue performs CPU tests, a W&B roundtrip, and per-worker B200 native
validation before starting production. Native uses first-block seeds only for
cross-checking; production freshly initializes the separately assigned second
block seeds. Runtime is isolated Python 3.12/JAX 0.6.2/Flax 0.10.7/Optax 0.2.5,
with pinned core/CUDA requirements. Record actual driver/runtime/device metadata.
No assumption of bitwise equality across GPUs or long production trajectories.

## Failure / reconnection policy

One learner process per tmux session, only assigned UUID in CUDA_VISIBLE_DEVICES.
No preemption, automatic worker retry, seed replacement, checkpoint resumption,
budget change, estimator substitution or numerical clipping. Preserve every
failed attempt and checkpoint. If the observer exits, start it with the same
state; it skips existing worker tickets. Never clear tickets or change state
directories to “retry”. Native/preflight infrastructure repair may be possible
before a production attempt, but inspect, preserve and explicitly record it.

Successful checkpoints use W&B COMMITTED manifest MD5/size/SHA metadata checks
before their own staging copies are released. Failure files remain. Results
are uploaded by the observer as hash-indexed artifacts; the original host
independently imports them through W&B (no VPN connection required).

If the learner exited zero and only administrative finalization failed,
`run.sh finalize --job <exact-name>` is read-only with respect to training.
Then restart/reconnect the observer to publish. A failure after nonzero learner
exit is NOT eligible for this repair. Keep healthy other workers running.

No need to copy OpenCode sessions/databases or configure model credentials.
Use the provider/model available to the user; never guess a remote GPT model
ID. Do not change OpenCode/global permissions to make experiment tools run.
Ask only for scientific/protocol decisions or access problems you cannot resolve.
