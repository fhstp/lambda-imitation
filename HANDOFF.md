# Handoff — offline Battleship memory investigation (2026-09-20 → 09-22)

Branch: `fix/retrace-lambda-critics`. Everything below is on it.

## What this was

The online Battleship runs never form board memory (`ship@fired` AUROC ≈ 0.54 =
chance). The hypothesis was a chicken-and-egg: memory only pays off once the
policy hunts, the policy only hunts once memory works. So we cut the policy half
out — a scripted **Bayes-density player** fills the replay buffer, the agent
takes 50 000 gradient steps on it with **zero environment interaction**, and a
probe asks what the recurrent memory encodes.

That worked on 5×5 and did not scale to 10×10, and chasing why uncovered four
genuine bugs in the λ-critic machinery. All four are fixed; none of them moved
retention on 10×10.

## TL;DR

1. **Offline expert data solves the memory problem on 5×5**: retention 0.54
   (online, chance) → **0.99**. The blocker was the data distribution, not the
   objective.
2. **On 10×10 retention is stuck in a 0.68–0.81 band** across 15 runs, against an
   untrained-FE anchor of 0.72 (bayes rollouts). Every estimator fix, LR setting
   and gradient-routing knob lands in that band.
3. **Four real bugs found and fixed** (below). They were worth fixing — the
   λ-critics were learning nonsense — but none changed retention.
4. **The bottleneck is the critic's per-action ranking, not the memory and not
   the actor.** Critic-greedy play (`argmax_a Q`) is as bad as actor-greedy play:
   both ≈ 90–96 shots where the Bayes player needs 50 and uniform random 94.5. The
   probe can read the board out of the FE; `Q(s,·)` never turns it into "shoot
   next to that hit".
5. **Prioritised replay at a sharp setting gave the best 10×10 retention**
   (0.810 / 0.807), though with the worst value behaviour of any run — see the
   caveat in the results table.

## How to read the metrics

- **Retention = `fired AUROC`**: ship-vs-water at cells already fired at. The
  memory saw the outcome and must still hold it.
- **Inference = `unfired AUROC`**: cells never fired at. *Never moved in any run*
  (balanced recall 50–57 % throughout).
- **Use balanced recall, not raw accuracy.** Ships are 14 % of a 10×10 board, so
  "all water" scores 86 %.
- **Anchor = update 0**, an untrained GRU: **0.646 / 53.2 % (actor rollouts),
  0.716 / 61.5 % (bayes rollouts)**. Not 0.5 — a random projection of the action
  history already decodes this much. Always compare against the anchor.
- **Probe budget matters.** In-training probe-evals use 20k collected steps +
  100k SGD; the final probe uses 100k + 500k and reads systematically higher, by
  0.05 in most runs and by 0.12 in one. **Compare curve-to-curve and
  final-to-final, never across.** A flat cheap curve has twice ended in the best
  final number.
- **Return scale**: `return = rows*cols + 1 − shots`. On 10×10 the Bayes player
  takes 50.4 shots (return 50.6), uniform random 94.5 (6.5). The learned policy
  has been at ~95 shots in every single run.
- **Value sanity**: with γ = 0.99 the analytic 1-step fixed point of this data is
  **+35.6**, the buffer's own MC discounted return averages **+55.5**, and the
  floor for any policy is ≈ −26 (legal-only) to ≈ −100 (stalling to the 1000-step
  env cap). Any `E[Q]` below −100 is impossible and indicates a bug.
- **`ld` split**: `ld_mean` is the constant offset between the λ-heads,
  `ld_std` the state-dependent part. **Only `ld_std` is signal.** Healthy runs
  have `std ≳ |mean|`; the pathological ones ran `mean` 50–100× `std`.
- **PER health**: `per_ess` 0.4–0.8 with priority CV (`per_priority_std /
  per_priority`) ≳ 0.5. ESS ≈ 1.0 with CV ≈ 0 means prioritisation is inert.
  ESS < 0.15 means the batch collapsed onto a few windows.
- **Extraction check**: `cg_return` / `cg_steps` (critic-greedy) versus the
  actor's, plus `qstd` / `qrange` (per-action spread of Q). This is the
  memory-vs-extraction splitter.

## Bugs found and fixed

### 1. V-trace applied to a Q-critic (fixed: `retrace=True`)

`loss_vtrace_lambda_sequence` regressed `Q(s, a_data)` onto a V-trace *state
value*, and V-trace multiplies the current step's delta by ρ. With a
near-deterministic actor against expert data, **ρ = 0 on ~97 % of transitions**
(measured: median π(a_data) = 0.000, ratio > 0.1 in 1.9 % of steps), so the
target degenerated to `v_s = V(s)` — no reward, no bootstrap, nothing pinning the
value. Retrace keeps the `k = t` term outside the trace product (coefficient 1),
so cut traces only remove multi-step propagation. `retrace_targets` is
module-level and unit-tested against a hand-rolled recursion.

### 2. λ-critics trained through `min(q1, q2)` (fixed unconditionally)

`get_q` returns the twin minimum, so gradient reached only the lower branch. The
branches drift apart (twin gap 0.285 → 2.432 over 150 updates; output-bias spread
0.21 for λ vs 0.004 for the SAC critic, which regresses both). Now both branches
are regressed against the shared target, as `loss_critic` always did. `min()`
belongs in the target, not in what is trained. **This changes results, so runs
from before this fix are not numerically comparable.**

### 3. `lambda_coef = 1.0` is far outside the reference regime (fixed: `--lambda-coef`)

The reference implementation (`~/git/lambda_discrepancy`, `batch_run_ppo.py:189`)
uses a **convex combination**, `ld_weight·LD + (1−ld_weight)·value_loss`, sweeping
`{0, 0.125, 0.25, 0.5}` and selecting 0.5 for Battleship-10. Ours was additive at
1.0 with no downweighting of the value losses. Sweep with the FE trained by the λ
objectives only, 12k updates:

| λ_coef | E[Q] λ₁ / λ₂ | ld (mean / std) | critic loss |
|---|---|---|---|
| 0.0 | +83.6 / +82.8 | 3.6 (0.8 / 1.7) | 0.37 |
| 0.01 | +80.9 / +81.2 | 4.4 (−0.3 / 1.9) | 0.58 |
| 0.1 | −24.7 / −24.7 | 0.7 (−0.05 / 0.8) | 62.2 |
| 1.0 | −113.9 / −60.2 | 2900 (−53.7 / 0.7) | 24.9 |

The cliff is between **0.01 and 0.1**. Note that at 0.1 the discrepancy is
*smallest* while the values are wrong — low `ld` is not evidence of health; the
term wins by distorting features until the heads agree.

### 4. The λ value level is not set by its own recursion

The decisive experiment. Same data, same 12k updates, only the encoder's training
signal differs:

| FE trained by | E[Q] λ₁ / λ₂ | ld |
|---|---|---|
| nothing (`--fe-lr 0`) | **+97.6 / +93.7** | 24 |
| everything (default) | −110 | ~400 |
| λ₁+λ₂+LD only (coef 1.0) | −113.9 / −60.2 | 2900 |
| λ₁+λ₂+LD only (coef 0.01) | +80.9 / +81.2 | 4.4 |
| all, coef 0.01 | −71 (at 25k, still falling) | 1–8 |
| all, coef 0.01, `--stop-actor-fe` | **+77 (stable)** | 9 (std > mean) |

So both are true: the LD term destabilises the encoder by itself at coef ≈ 1,
**and** the actor's gradient into the shared FE drags the λ value level down
regardless of coefficient. `--stop-actor-fe` + `--lambda-coef 0.01` is the first
configuration where the λ-critics converge to a sane value (+77 against a +35.6
fixed point and +55.5 MC return) with critic loss < 1 and a
state-dependent-dominated discrepancy.

## Flags added (all default to previous behaviour)

| flag | default | recommended | what |
|---|---|---|---|
| `--retrace` | off | **on** | Retrace(λ) targets for the λ-critics instead of V-trace |
| `--lambda-coef` | 1.0 | **0.01** | weight on the λ-discrepancy term |
| `--ld-center` | off | — | minimise `var(Q1−Q2)` instead of `mean((Q1−Q2)²)`; no measured effect once the other fixes are in |
| `--stop-critic-fe` | off | — | detach the SAC critic from the FE (λ side keeps the live latent). Worst retention of any run |
| `--stop-actor-fe` | off | **on** | detach the actor from the FE. Required for stable λ values |
| `--per-alpha` | 0.0 | **1.5** | prioritised replay exponent |
| `--per-beta` | 0.4 | 0.4 | IS-weight exponent (weights normalised by batch max) |
| `--per-window` | 0 | **3** | window *prefix* the trace-mass priority is computed over. **0 = whole truncation span is useless** — the grounded fraction concentrates and the score is flat (ESS 1.00) |
| `--per-ratio-floor` | 1e-3 | 1e-3 | lower clip on each ratio in the priority |
| `--actor-critic` | `sac` | — | extract the policy from `lambda1` / `lambda2` instead. Makes that critic inflate (see below) |
| `--critic-greedy-eval` | **on** | on | log critic-greedy return/steps and `qstd`/`qrange`. Read-only |
| `--setup-only` | off | — | build env/agent/probe helpers and exit; used by `battleship_offline_bayes.py` |

Library-level (no CLI): `fns.prefill_buffer(behaviour_fn=…)` fills the buffer
from a scripted policy, `fns.update_only(state, n, key)` runs env-free updates.

## Results — 10×10, ships (5,4,3,2), 100k Bayes transitions, 50k updates

Final probe (100k collect + 500k SGD). Anchor **0.646 / 53.2 %** actor,
**0.716 / 61.5 %** bayes.

| # | run | actor AUROC / bal | bayes AUROC / bal |
|---|---|---|---|
| 1 | V-trace, lr 1e-4 (original baseline) | 0.725 / 61.7 % | 0.781 / 68.0 % |
| 2 | V-trace, lr 1e-4, ε=0 data | 0.678 / 58.8 % | 0.763 / 67.0 % |
| 3 | V-trace, **all lr 1e-5** | 0.765 / 60.0 % | **0.830** / 69.7 % |
| 4 | V-trace, fe-lr 1e-5 only | 0.664 / 50.5 % | 0.780 / 58.9 % |
| 5 | V-trace, `--stop-critic-fe` | 0.655 / 52.0 % | 0.701 / 54.4 % |
| 6 | Retrace, lr 1e-5 | 0.749 / 56.5 % | 0.829 / 69.3 % |
| 7 | Retrace, lr 1e-4 | 0.770 / 64.1 % | 0.774 / 67.1 % |
| 8 | Retrace + twin fix, lr 1e-4 | 0.720 / 61.5 % | 0.748 / 66.3 % |
| 9 | #8 + PER (α 0.6, w 5) | 0.740 / 62.9 % | 0.759 / 66.9 % |
| 10 | #8 + **PER (α 1.5, w 3)** | **0.810 / 68.8 %** | **0.807 / 71.7 %** |
| 11 | Retrace + twin + coef 0.01 + `--stop-actor-fe` | 0.684 / 59.9 % | 0.746 / 66.2 % |

**Run 10 is the best retention and the worst value behaviour** — E[Q] −138, `ld`
up to 10⁴, and its in-training probes were the lowest of any run (0.58–0.64,
below anchor) before the final probe read 0.810. Its PER diagnostics were healthy
(ESS 0.30–0.67, priority CV ≈ 1.4), so the prioritisation did bite. Whether the
0.810 is the prioritisation or probe-budget noise **needs a second seed** before
anyone builds on it. Note runs 1–7 predate the twin fix.

5×5, ships (3,2), for contrast — same code, same objective:

| data | actor | bayes |
|---|---|---|
| ε = 0.1 | 0.964 / 84.5 % | 0.926 / 79.1 % |
| ε = 0 (greedy) | 0.941 / 80.0 % | **0.993 / 94.5 %** |

## The actual bottleneck: the critic has no action ranking

From the `--actor-critic lambda1` run (first with critic-greedy logging):

| updates | actor ret / steps | critic-greedy ret / steps | qstd | qrange |
|---|---|---|---|---|
| 1k | 4.8 / 96.2 | 15.7 / 85.3 | 1.87 | 10.3 |
| 10k | 8.0 / 93.0 | 7.9 / 93.1 | 2.77 | 14.0 |
| 20k | 5.0 / 96.0 | 5.6 / 95.4 | 3.27 | 17.8 |

Critic-greedy plays the same as the actor, both ≈ random (94.5), against the
Bayes player's 50. And the **best** critic-greedy play in the run is at 1k
updates, before the critic learned anything — training makes the value scale
correct and the ranking *less* useful.

Direct profile of the SAC critic on a trained agent, 2000 states:

```
Q min legal 98.88   Q mean legal 104.44   Q max legal 111.11   (values ≈ +105)
Q(s, a_data) 105.77            Q(s, a_actor) 109.78
```

A 12-unit spread across 100 legal actions, i.e. the action signal is ~10 % of the
value scale — and +105 is above the +100 return ceiling, so it is overestimating
on actions the data never took.

One mechanistic note: with `--actor-critic lambda1`, λ₁ inflated relative to λ₂
(74.1 vs 67.5, reversing their prior agreement) and `ld` went offset-dominated
again. **Whichever critic the actor maximises acquires the OOD overestimation** —
so "just extract from the calibrated critic" does not work; the calibration was a
property of not being maximised.

## What I would run next, in order

1. **Second seed on run 10** (PER α 1.5, w 3). It is the best number and rests on
   one seed with contradictory in-training probes. Cheap, decisive.
2. **Attack the advantage structure**, since extraction is exonerated:
   - `--no-critic-layer-norm` (queued locally; LN plausibly washes out a 10 %
     action signal),
   - a dueling head, `Q(s,a) = V(s) + A(s,a)` with `A` mean-centred, so the
     per-action signal is not competing with the value scale,
   - an auxiliary per-action target (e.g. predict P(hit) per cell) to force the
     head to use the board information the probe proves is in the FE.
3. **Entropy floor** — the actor sits at 0.13–0.45 nats against ln(100) = 4.61,
   committing to one cell per state while the Q differences are near-noise.
   `--autotune-alpha --target-entropy 2.3` is queued locally.
4. **Anchor the policy to the data** (BC/AWAC term). Fixes the ρ = 0 collapse at
   its source, would make the λ-discrepancy meaningful (with cut traces λ₁ and λ₂
   see identical targets, so the discrepancy is *structurally* zero), and gives
   the actor something to learn from — it never improved in any run.
5. **A full-probe anchor**: run an untrained FE through the 500k-step probe. The
   0.646/0.716 anchor is on the cheap probe, so there is currently no absolute
   floor under the final column.

## Gotchas

- **PER requires `fake_onpolicy_loss=False`** (it pins every ratio to 1, making
  all priorities identical). `create_iqlearn` now raises instead of silently
  sampling uniformly. The probe script already sets it False.
- **`--per-window 0` is inert.** Use 3–8.
- **The twin fix changes results** — runs before it are not comparable.
- **Old checkpoints still load.** `Buffer.priorities` defaults to `None` and adds
  no pytree leaf unless `per_alpha > 0`. An agent saved *with* PER cannot be
  loaded into a config without it (leaf-count mismatch) and vice versa.
- **`terminal_bonus`** is only passed to the env when `--terminal-bonus` is set;
  older `lambda-envs` (including the local clone) has no such kwarg, and the
  script exits with a clear message rather than a `TypeError`.
- **`q`, `v`, `entropy` metrics come from `loss_actor`**, i.e. they are the
  *actor's* values (`Σ_a π(a)Q(s,a) + αH`), not the data actions'. The λ metrics
  (`lambda{λ}_critic:`) are at the data action. Do not compare them directly —
  that mistake cost an hour.

## Repro

```bash
git checkout fix/retrace-lambda-critics && pip install -e ".[dev]"

# the best-retention config (run 10)
python examples/lambda-envs/battleship_offline_bayes.py \
  --rows 10 --cols 10 --ship-lengths 5,4,3,2 \
  --retrace --per-alpha 1.5 --per-beta 0.4 --per-window 3 \
  --updates 50000 --offline-epsilon 0.1 --probe-eval-interval 10 \
  --output-dir <out>

# the healthiest-value config (run 11)
python examples/lambda-envs/battleship_offline_bayes.py \
  --rows 10 --cols 10 --ship-lengths 5,4,3,2 \
  --retrace --lambda-coef 0.01 --stop-actor-fe \
  --updates 50000 --offline-epsilon 0.1 --probe-eval-interval 10 \
  --output-dir <out>

# 5x5, where this all works
python examples/lambda-envs/battleship_offline_bayes.py --offline-epsilon 0
```

~45–50 min per 10×10 run on an RTX 3090. Artefacts land in
`<out>/{actor,bayes}/` (datasets, probe params, episode/accuracy/retention PNGs)
plus `offline_history.pkl` with the full per-round metric history and the
probe-eval curve.

Tests: `mamba run -n lambda pytest tests/` — 121 passing.
