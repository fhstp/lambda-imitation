# Fresh occupancy/correction study — predeclared protocol

The user authorized proceeding with implementation and validation on 2026-11-13.
Production requires the separate hash-bound admission receipt; this document
alone is not permission to bypass a failed or missing gate. All old
history/replay-correction results, source seals, environments and manuscript
snapshots remain untouched. New results must not replace or pool historical
cohorts. Driver/runtime installation is not part of this authorization.

## Two questions, separated

1. Does episode-discounted starting-state training change practical learning
   relative to a matched uniform-start control?
2. With a genuinely fixed behaviour, target actor and representation, do
   uncorrected, ordinary per-decision IS and Retrace fits use the specified
   discounted starting measure, and how do their learned discrepancies differ?

Adaptive online training cannot certify the fixed-policy theorem. A discounted
empirical mixture of adaptive episodes is not called one stationary behaviour's
occupancy posterior. Sampling alignment is also not proof that finite neural
critics have reached an exact fixed point or that their representation is Markov.

## A. Fresh end-to-end factorial

Two starting measures (`uniform`, `discounted`) crossed with three future trace
coefficients (`uncorrected`, `is`, `retrace`). Six conditions per environment:

- Uniform: direct uniform anchors over retained complete-episode transitions.
- Discounted: direct anchors with global probability proportional to gamma**age.
- Uncorrected: c=lambda.
- Ordinary IS: c=lambda*pi_target/mu_logged, no ratio clipping/self-normalization.
- Retrace: c=lambda*min(1,pi_target/mu_logged).

Both measures use exactly the same episode retention, reconstruction and losses.
The uniform control is NOT the historical contiguous-window implementation.
These changes relative to the paper are disclosed, not attributed exclusively
to discounting by comparing to a historical curve.

| Setting | Battleship | Minesweeper |
| --- | --- | --- |
| Independent seeds per condition | 10 | 20 |
| Five-seed jobs per condition | 2 | 4 |
| Training seeds | 2026111300..2026111309 | 2026111400..2026111419 |
| Online calls / seed | 500,000 | 1,000,000 |
| Random-prefill calls / seed | 12,800 | 6,016 |
| Rounds / online calls per round | 50 / 10,000 | 200 / 5,000 |
| Primary endpoint including prefill | 492,800 | 1,006,016 |
| Greedy eval episodes / checkpoint | 50 | 128 |
| Evaluation cadence | Every round | Initially and every round |
| GRU / embedding / head width | 512 | 128 |
| Paper sequence batch / learning length / tail | 128 / 40 / 50 | 64 / 32 / 32 |
| Scored actor / task-discrepancy / auxiliary anchors per update | 11520 / 11392 / 5120 | 4096 / 4032 / 2048 |
| Reconstructed episodes per update | 512 | 256 |
| Lambda pair / discrepancy coefficient | (0.05,0.85) / 0.01 | (0,0.95) / 0.1 |
| Feature / actor / critic learning rates | 1e-5 / 1e-4 / 1e-4 | 1e-5 / 1e-4 / 1e-4 |
| Entropy coefficient | 0.1 | 0.01 |
| Discount / EMA rate | 0.99 / 0.005 | 0.99 / 0.005 |

All auxiliary/task regression is MSE; twin branches regress separately. Actor
gradients stop at recurrent features; discrepancy gradients stop at auxiliary
heads. Bootstrap values use target networks and reward-only residuals. Collection
probabilities are recorded from the actual online policy, with real action masks.

The original numbers of scored positions are preserved, now as explicit anchor
draws rather than window positions. A batch reconstructs four times the paper's
sequence batch in complete raw episodes, reusing features for multiple anchors.
Each anchor has the right marginal law; shared-episode anchors are correlated,
not independent observations. This is a declared implementation change. Full
episodes replace zero-start window burn-in and finite context; no new capacity
or hyperparameter search is performed.

Replay admits only complete episodes and removes complete oldest episodes to
stay at or below 100,000 retained transitions. Pending raw episodes survive host
round and prefill boundaries. No artificial terminal or free extra trajectory
completion is allowed at budget boundaries. Pending calls are counted even if
not yet sampleable. No learned carry from previous parameter versions is used to
reconstruct replay; collection also reconstructs the current raw prefix.

The factorial comprises 36 jobs / 180 independent training seed instances.
Matched numeric seed labels across arms support paired exploratory contrasts;
they do not multiply the count of independent training seeds within an arm.

## B. Fixed-policy/representation critic controls

For each seed, use the NEW uniform-start/Retrace learner's primary-endpoint
encoder and online actor as the predefined common reference. Selection does not
depend on return, discrepancy, stability or any other measured ranking.

Freeze that encoder and actor and generate a separate fixed-behaviour dataset:
uniform over legal actions for Battleship, uniform over all actions for
Minesweeper. The collector never calls the learned actor and never changes its
law. Replay begins at the environment's real initial law, has complete episodes,
and records exact collection probabilities. Both games' finite timeout/terminal
conventions are explicit. Battleship's full legal-action mask remains available
to the policy and is part of the declared observation context.

Use the SAME dataset, frozen reference, held-out diagnostic dataset and fresh
auxiliary initialization across the six sampler/trace fits for each seed.
Only the four auxiliary branches and their EMA targets are fitted; encoder,
actor and task critic must retain identical parameter hashes before/after.
For each fit, use the corresponding paper architecture, batch/loss counts,
MSE and critic learning rate. The shared fitting dataset receives exactly
100,000 real environment calls per seed and the independent held-out dataset
10,000. Keep their final incomplete episodes unscored and account for those
calls explicitly. Each fresh auxiliary fit receives exactly 12,000 updates,
with diagnostics initially and every 1,000 updates. These supplemental budgets
are fixed before production; none is selected from a production result.
Validation seeds are disjoint from production. No tuning from production audit
outcomes is allowed. Both datasets are generated only once per reference seed
and shared byte-for-byte across the six fits, not recollected separately per arm.

This tests the starting-measure convention on the real environments; it is not
an exact finite-state value solution for their large neural representations.
Independent finite-model controls (including genuine observation aliasing at
different episode ages) test the operator and posterior identities exactly.

## Statistics, diagnostics, and recovery

Use per-training-seed returns/discrepancies, then pointwise 95% Student-t CIs.
Report all seeds, predeclared endpoints, actual calls, and failures. No selected
peak or survivor-only mean. Pair within-study contrasts by nominal seed, describe
them as exploratory without multiplicity adjustment. Sampled-policy discrepancy
and greedy return are distinct measures; label their visitation laws explicitly.

Log actual selected episode ages, retained and pending counts, collector policy
versions, ratio/target magnitudes, finiteness and checkpoints. Future traces,
starting-state selection, and evaluation weighting must have separate names.

Use persistent state on /raid, unique IDs, one learner/GPU/tmux session, and a
persistent observer. Never reuse historical assignment state. Preserve all
failed attempts. Administrative observer/upload repair may proceed without
retraining. Genuine numerical failures are scientific outcomes, not permission
to clip, change an estimator, replace a seed, or retry until success. Any
training restart/resumption requires a separately recorded decision and physical
call accounting. Host-level repair still requires user approval.

## Admission gates

- Independent posterior/occupancy and forward-trace operator checks.
- Actual collection, probability logging, episode retention and call accounting.
- Actual anchor-only losses, gradient routing, frozen-network identity checks.
- Unchanged-environment CPU regression tests and full-shape B200 checks for each
  environment/sampler/trace path; native speed/resource measurement.
- Complete predeclared production/audit budgets and seeds, immutable source
  manifest, W&B authentication/source/checkpoint canary, queue lifecycle tests.

Until these gates pass and the remaining audit budgets are fixed, no production
job may be started. None of these preflight tests admits historical-source edits
or a claim that mixed adaptive replay satisfies a stationary-policy theorem.
