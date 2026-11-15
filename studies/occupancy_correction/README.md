# Fresh occupancy/correction study

This is a separate study requested after the B200 history jobs completed.
Production starts only with a passing, hash-bound admission receipt. Read live
state rather than inferring completion/admission from these instructions.
No driver/package changes are needed. Use the existing isolated Python runtime;
**do not run the historical deployment queue from this worktree.**

## Sampling and training

`sampling.py` implements a genuinely different replay-anchor distribution:

```
q_discounted(episode, t) = gamma**t / sum_all_valid_transitions(gamma**age)
```

It draws actual anchors from that law, with zero probability on padding, and
returns the entire raw episode so a learner can reconstruct the true prefix and
future. A matched `uniform` law samples valid transitions uniformly. This control
is **not** claimed to reproduce the legacy contiguous-window sampler.

Episode-local normalization is deliberately avoided. Datasets containing more
than one declared behaviour policy are rejected for this fixed-behaviour path.
Policy identifiers still require real collection provenance; labels alone do
not establish stationarity or the theorem's observation/action-law assumptions.

The learner scores only the sampled anchors. Scoring all positions in the
returned episodes would change the measure again. Sampling does not apply or
replace action-probability importance corrections: those separately determine
the future TD-residual coefficients (`lambda`, `lambda*rho`, or
`lambda*min(1,rho)`).

`replay.py` implements whole-episode FIFO retention and a factored sampler with
the same global law. `collection.py` logs the actual collection probabilities
and preserves pending episodes across round/budget boundaries. `learner.py`
reconstructs current/target features from real episode starts, uses anchor-only
losses, and separately implements adaptive learning and frozen critic fits.

`PROTOCOL.md` explains the scientific scope and all departures from legacy window
replay. Adaptive episode mixtures are not claimed to be one stationary mu.
The separate fixed-policy fits use one frozen reference and a byte-identical
fixed-uniform fitting/held-out dataset across the six fits for each seed.

## Independent validation

`test_sampling.py` includes:

- A variable-length episodic Markov model solved by an independent NumPy
  discounted resolvent, with aliased states whose posterior distinguishes
  uniform, correct discounted, and incorrectly episode-normalized sampling.
- Actual empirical draw-frequency and conditional-posterior checks, including
  zero probability for incomplete/padded positions.
- Preservation of true raw prefixes, real terminals, and logged behaviour
  probabilities, with rejection of incomplete episodes and behaviour mixtures.
- Sampled-anchor multi-step targets checked against independent forward
  products in all three correction arms, including ratios greater than one.
- A check exposing the wrong loss measure if a consumer scores every returned
  position instead of only the sampled anchors.

Additional tests exercise actual collection, retention, losses, gradient routing,
fixed-network identity, serialization, all worker paths, allocation, and recovery.
`test_operator.py` independently enumerates an aliased finite POMDP and solves its
matrix operators. It distinguishes uniform/discounted posteriors and verifies
ordinary IS, Retrace, and uncorrected forward traces. It also verifies that
target/proposal occupancy importance weighting and direct discounted sampling
give the same expectation in that fixed-behaviour model.

On 2026-11-13, all **27 CPU tests passed** using the unchanged `.venv-b200`
runtime from the original checkout. `sampler_validation.json` records their
scope, source hashes, and JUnit report location. All 44 historical sealed files
also retained their original hashes. This receipt is not production admission.

## Production gates

Before any production learner starts:

1. Agree and record the scientific scope, fixed behaviour collection law,
   representation/target-policy treatment, and any adaptive-learning controls.
2. Implement and validate actual collection, whole-episode replay retention,
   recurrent prefix reconstruction, anchor-only losses, and physical-call
   accounting. Respect genuine environment terminals and budget-edge censoring.
3. Preserve paper-matched independent seed counts (10 Battleship, 20 Minesweeper
   per condition), explicit interaction/evaluation budgets, and all failures.
4. Validate end-to-end full-shape GPU smoke runs without reusing their state for
   production; fix the source and protocol hashes before admitting workers.
5. Assign unique run IDs and one tmux learner per assigned GPU, plus persistent
   read-only monitoring. Define recovery without silent retuning, clipping,
   seed replacement, or success-only retries.

In particular, the sampler must not be wired into adaptive mixed replay and
then advertised as the posterior of one fixed stationary behaviour. Online
learning curves and fixed-policy/frozen-representation theory controls must be
identified separately.

## Operational paths and inspection

- Checkout: `/raid/ustp/git/lambda-occupancy-correction-2026-11-13`.
- Python: `/raid/ustp/git/lambda-imitation/.venv-b200/bin/python` (unchanged).
- State: `/raid/ustp/experiments/occupancy-correction-2026-11-13`.
- Planned allocation: 36 learning jobs, six shared-dataset jobs and 36 fixed-fit
  jobs, with five seeds per job. Ten Battleship / twenty Minesweeper seeds per
  condition. There is no training-resume command.
- Each worker uses its own `oc-*` tmux session and only one GPU UUID. The
  persistent administrative observer uses `oc-observer-20261113`.
- Per-job configuration, metrics, checkpoint/upload receipts, status and logs
  are under `STATE/jobs/JOB/`. `STATE/study.json` and `STATE/admission.json` bind
  allocation, source and validation. Successful checkpoints release only their
  own staging copy after remote COMMITTED, size, MD5 and SHA metadata checks.
  Failure files, references and shared datasets remain local.
- `queue.py status` reads current job/tmux state without launching workers.
- Reconnecting the observer to the SAME state skips consumed tickets; it can
  repair administrative uploads/finalization, never retry a learner.
- `report.py` writes new study reports only. It refuses inferential summaries
  unless every planned seed block has completed, and never edits the paper.

W&B projects are `retrace-rebuttal-occupancy-battleship-20261113` and
`retrace-rebuttal-occupancy-minesweeper-20261113` in the existing research entity.
Validation/source/checkpoint/result artifacts are coordinated through
`retrace-rebuttal-validation`. Historical projects are not modified.
