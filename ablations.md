# Porting the λ-discrepancy memory mechanism off-policy — ablation study

**Question.** The λ-discrepancy (Allen et al. 2024) forces a recurrent agent to build
Markov memory by minimizing the disagreement between two value estimates computed at
different bootstrap horizons (λ). It was demonstrated with **on-policy recurrent PPO**.
Does it — and does memory formation in general — survive a port to an **off-policy**
(replay-based, soft-Q / IQ-Learn) setting, en route to off-policy imitation?

**Probe task.** Battleship 10×10 (ships 5,4,3,2). The agent only ever observes
`[last_hit_miss (1) | legal_action_mask (100)]` — a partial observation: it sees whether
its *last* shot hit and which cells remain unfired, but never a map of past hit locations.
Recovering the board therefore *requires* recurrent memory.

**Metric.** We train a decoder (MLP) on the frozen recurrent carry to predict ship
occupancy, and report **order-invariant `fired_auroc`**: AUROC of hit-vs-miss among *fired*
cells, evaluated on rollouts whose fire order is randomized (ε=1). Randomizing the order is
essential — under a deterministic policy the carry can encode a *positional "tape"*
(cell ≡ timestep) that inflates a fixed-order probe without any spatial board model. The
order-invariant probe measures genuine spatial memory.

**Targets.** The paper's on-policy recurrent PPO reaches **fired_auroc ≈ 0.99**
(order-invariant; genuine). Our off-policy port sits at **≈ 0.55 (chance)**. This document
is the bisection of that gap.

> Convention: all `fired_auroc` values below are the order-invariant (ε=1) metric.
> "Our stack" = the off-policy soft-Q / IQ-Learn codebase; "reference" = the original
> lambda-discrepancy recurrent-PPO codebase, re-run and probed with the *same* harness.

> **Answer: see Part J.** Parts A–I bisect *backwards* from our stack and return mostly nulls;
> Part J builds the off-policy agent *forwards out of* the reference and localises the failure to
> a two-component interaction (per-action Q regressed only at the executed action × a
> near-uniform actor). Parts A–I remain as the record of what was eliminated, including three
> conclusions Part J corrects.

---

## Part A — Off-policy component bisection (our stack)

Starting from the failing off-policy end-to-end run (`fired_auroc ≈ 0.55`), we varied one
component at a time. **None recovered memory.**

| Ablation | Setting(s) | fired_auroc | Verdict |
|---|---|---|---|
| Replay staleness | buffer 200k / 20k / 5k | ~0.55 (all) | staleness not the cause |
| Gradient reuse | batch 4 / 16 / 128 | ~0.55 (all) | reuse not the cause |
| Critic head | per-action twin-Q vs scalar-V | ~0.55 (both) | Q-vs-V not the cause |
| Bootstrap horizon | λ₁ = 0.1 vs 0.7 | ~0.55 | value diverges either way |

**Infrastructure issues found & fixed along the way** (all additive, default-off flags):

- **No gradient clipping existed** in the off-policy code (optimizers were plain Adam) while
  the reference clips global-norm to 0.5. Added `--grad-clip`.
- **`fake_onpolicy_loss` defaulted to True** — every V-trace importance ratio was clamped to
  1, silently disabling the off-policy correction. This alone drove the low-λ value to the
  no-termination fixed point −1/(1−γ)=−100 (deadly triad, uncorrected).
- **The +100 terminal reward destabilizes our off-policy twin-Q** (critic loss 82→641, PG
  −4e8) even with grad-clip; `terminal_bonus 0` (−1/step) is stable. (Reference is stable at
  +100 because on-policy.)

---

## Part B — The actor

| Ablation | Observation | Fix |
|---|---|---|
| Unclipped V-trace PG | entropy collapses 0.28→0 in ~4 steps; `pg_loss` explodes −117→−45M; policy collapses to a deterministic, board-blind firing order | `--ppo-clip-eps` (PPO ratio clip = trust region) |
| Alpha-autotune (target H=1.5) | entropy still →0 **and** α runs away to 100 (integrator windup); unclipped PG wins the race | rejected — not a fix |
| PPO-clip actor | stability solved (entropy holds), but policy **frozen** at ~uniform | — |
| + advantage normalization `(Â−mean)/std` | still frozen | — |

**Finding.** The clip fixes the blow-up, but the policy still doesn't improve: under
random play with a −1/step reward, a single action's quality is invisible (only the
long-horizon steps-to-clear signal exists), so the per-action advantage is ≈noise. A
frozen/blind policy never needs board memory. This is a **chicken-and-egg**: spatial
memory ⟸ targeting policy ⟸ advantage signal ⟸ spatial memory.

---

## Part C — Data quality and value stability

To break the chicken-and-egg we injected a **skilled ε-greedy hunt/target expert** as the
behaviour policy (reads only observable `hits_misses`; verified: clears 10×10 in ~61 shots
vs ~94 random, highest return variance at ε=0.3).

| Ablation | Observation |
|---|---|
| Skilled data, real IS (ρ<1) | value **drifts to −100** (v starts correct at ~−58, then diverges) |
| Skilled data, ρ=1 (fake_onpolicy) | value **explodes to +200**, discrepancy ld → 6e4 |
| Prefill bug | `prefill_buffer` ignored the behaviour policy → the large warm-up buffer was random-contaminated during exactly the early rounds. Fixed (prefill now honors the expert). |
| Huber discrepancy | `loss_scalar_v_ld` used raw squared `(V₀−V₁)²` (unbounded gradient) vs the reference/docstring's **Huber**. Fixed → bounds ld, but the low-λ head still explodes via shared-FE coupling |
| `--scalar-v-bound` (v = −B·σ(raw)) | value can no longer explode (∈(−B,0)); ld bounded; heads disagree — yet **auroc still flat (0.55)** through 60 rounds |
| Value-only (`--lambda-coef 0`, no discrepancy) | value is **stable** (held ~−28 / −15) | 

**Two key mechanistic findings:**

1. **V-trace importance-ratio collapse.** With a near-uniform online actor
   (π≈1/n_legal≈0.014) and a peaked expert (μ≈0.23), the IS ratio **ρ≈0.06**. V-trace weights
   the reward correction by ρ, so the value barely sees the rewards and bootstraps on itself
   → drift/divergence. *Corollary:* off-policy value learning off expert data requires the
   policy to match the expert (ρ≈1) — i.e. **imitation** — which is exactly the project goal.
2. **The λ-discrepancy loss is the destabilizer in our stack, not the value.** Value-only
   training is stable; adding the discrepancy makes the shared feature extractor diverge.
   With `--scalar-v-bound` the value is stable *and* the discrepancy is bounded and
   non-trivial, yet memory still does not form — the objective settles into a **trivial
   memoryless minimizer** (both λ-heads agree on a state-independent constant).

---

## Part D — Is board memory even *value*-relevant? (value-memory-gap diagnostic)

On rolled-out data we fit return-to-go predictors from nested feature sets (MLP, held-out):

| features | 10×10 expert R² | 10×10 random R² |
|---|---|---|
| memoryless obs (coverage mask) | **0.954** | **0.994** |
| + cumulative hit **count** (scalar) | 0.956 | 0.994 |
| + hit **locations** (spatial, = probe target) | 0.968 | 0.996 |
| + board / ship map (oracle, *unobservable*) | 0.998 | 0.999 |

**Finding.** The memoryless observation already explains ~95% of return variance; the
*reachable* spatial memory adds at most **+1.4% R²**. So board memory is **not
value-relevant** — no value-side objective (V-trace *or* the λ-discrepancy) has a real
incentive to encode it. It is **policy-relevant** (targeting needs hit locations), which
forms only when the policy improves.

---

## Part E — Reference ablation grid (the system that *works*)

We ran the reference and probed with the same harness, ablating each component we had
suspected. Modifications added to the reference harness: `FREEZE_POLICY` (zero the PG +
entropy loss), `LD_WEIGHT` (0 disables the discrepancy), `REPLAY_ROLLOUTS` (ring-buffer
replay of past rollouts), `RECOMPUTE_TARGETS` (recompute GAE/λ targets from the current net
each update, so the target *moves* — like our V-trace, vs the reference's fixed stored targets).

| Config | policy | λ-discrepancy | data / targets | fired_auroc |
|---|---|---|---|---|
| baseline | improving | on | on-policy, stored | **0.997** |
| ld-off | improving | **off** | on-policy, stored | 0.991 |
| frozen | **frozen** | on | on-policy, stored | 0.890 |
| frozen + ld-off | **frozen** | **off** | on-policy, stored | 0.920 |
| replay ×32 | improving | on | **replay-32**, stored | 0.997 |
| replay ×128 | improving | on | **replay-128**, stored | 0.999 |
| replay ×32 + recompute | improving | on | replay-32, **recomputed** | 0.999 |
| on-policy + recompute | improving | on | on-policy, **recomputed** | 0.994 |

**Finding — the reference forms 10×10 memory under *every* ablation (0.89–0.999):** with no
policy improvement, with no discrepancy, off replay up to 128 rollouts deep, and with
recomputed bootstrapped targets. Therefore **none** of policy-improvement, the
λ-discrepancy, off-policy replay/staleness, recomputed-vs-stored targets, or (episode-aligned)
stored-vs-zero carry explains our 0.55.

Two corollaries:
- **The λ-discrepancy is essentially a passenger for memory *formation*** — ablating it
  (0.997→0.991 improving; 0.890→0.920 frozen) barely moves the needle. It sharpens memory
  slightly but does not create it. (This does not speak to its role in more strongly partial
  tasks; on Battleship the recurrent value/policy already encode the board.)
- **Off-policy is not the barrier.** Replay-128 forms memory fine — so our 0.55 is a specific
  porting difference, not a fundamental off-policy obstacle.

**Observation confound ruled out.** Both codebases feed the FE a structurally identical
observation `[hit_bit | coverage_mask]` — same POMDP, valid apples-to-apples comparison.

---

## Part F — Where the gap actually is: FE capacity vs RL objective

**Supervised FE-capacity audit.** We trained *our* feature extractor (paper-arch GRU, H=512)
end-to-end (BPTT), with **no RL**, to predict the board from the carry, for two FE inputs:

| FE input | held-out fired_auroc |
|---|---|
| hit-bit-only `o[...,0:1]` + prev-action (**our RL FE input**) | **0.789** |
| hit + coverage mask (reference-like), 101-dim | 0.565 (overfit: train BCE 0.007, poor generalization) |

**Findings.**
- **Our FE is capable.** With the exact (minimal) observation our RL uses, supervised
  training reaches **0.789** order-invariant — the GRU *can* encode the 10×10 board. Not an
  architecture wall.
- The coverage mask *hurt* here (overfitting on only 400 training episodes) — refuting the
  hypothesis that stripping coverage in `obs_fn` was the cause. The 1-dim input actually
  generalizes better because it *forces* the network to use the hit signal.
- **The decisive gap: RL 0.55 ≪ supervised 0.789 on the same FE.** Our RL objective is
  failing to *extract* the board memory the FE is demonstrably capable of forming. The locus
  is the **RL training's shaping of the shared FE**, not FE capacity, not the observation,
  not the data, and not the discrepancy.

---

## Part G — Value-gradient parity bisection (matching PPO piece-by-piece)

If our value→FE gradient is what underperforms, we can bisect toward PPO's by neutralizing each
concrete difference, in the cleanest regime (tiny fresh buffer, no discrepancy, trust-region
V-trace actor, episode-aligned whole-episode windows, agent's own policy).

| Variant | vs PPO difference removed | fired_auroc |
|---|---|---|
| on-policy-like (real IS) | staleness (buffer 512/4096) | ~0.53 flat |
| ρ=1 (`fake_onpolicy`) | importance-weighting (GAE ignores ρ) | ~0.53 flat |
| bound-off (`--scalar-v-bound 0`) | sigmoid value bound | 0.52 flat |
| τ=1 (`--tau 1.0`) | EMA value-target lag (PPO has none) | 0.55 flat |

**None recovers memory.** Matching PPO's value gradient factor-by-factor — no staleness, no
importance weights, no value bound, no target-network lag — still yields chance. (The one
untested option is all of these *combined* + MSE-vs-Huber.) This strongly corroborates Part F:
the deficiency is not any single, easily-named term in the value objective.

**Recurrent representational drift (importance-ratio diagnostic).** We logged the actor IS ratio
resolved by position in the episode. The ratio is **≈1 at the window start** (t=0, zero carry:
`|log ρ|_t0 ≈ 0.00`) and **spreads with depth into the episode** (`|log ρ|_tlast ≈ 0.13–0.15`),
even though the policy *weights* barely move. Interpretation: re-rolling the recurrent carry under
slightly-updated weights makes the carry — and thus the action distribution — diverge more at
each of the ~95 steps (the R2D2 representational-drift effect). The clean t=0 ≈ 1 rules out a
bookkeeping/carry-mismatch bug, and the systematic mean-ratio < 1 seen earlier is its signature.
This is why off-policy importance ratios are wide despite tiny weight changes — though note it is
*not* the memory-blocker (ρ=1 already ignores the ratios and still fails).

---

## Synthesis

The through-line of the whole study:

1. On Battleship, **spatial board memory is policy-relevant, not value-relevant** (≤1.4%
   value R²), and the **λ-discrepancy is not what forms it** (reference forms memory with the
   discrepancy off). Memory arises from a recurrent network processing the observation stream
   under *any* reasonable objective — the reference does so robustly, even fully off-policy.
2. **Off-policy replay, staleness, reuse, target recomputation, and carry handling are all
   ruled out** as the cause of our failure (reference is robust to each; our own sweeps flat).
3. **Our FE can form the memory** (0.789 supervised) but **our RL objective does not extract
   it** (0.55). Every value-side pathology we chased — the −100 / +200 divergences, the
   ρ-collapse, the memoryless minimizer — is a symptom of the same root: our off-policy RL
   loss does not deliver a clean, strong memory-forming gradient to the shared FE, whereas the
   reference's does.

**Durable engineering deliverables** (all additive, default-preserving flags): `--grad-clip`,
`--ppo-clip-eps` (trust-region V-trace actor), `--vtrace-normalize-advantage`,
`--scalar-v-bound`, `--scripted-behaviour`/`--scripted-epsilon` (skilled expert +
`behaviour_fn` wired through collection *and* prefill), Huber λ-discrepancy for the scalar-V
head, `--online-batch-size`; plus reference-harness switches `FREEZE_POLICY`, `LD_WEIGHT`,
`REPLAY_ROLLOUTS`, `RECOMPUTE_TARGETS`.

## Part H — Root cause: the Huber × value-target-net interaction

Rather than accept "diffuse," we bisected from the *working* reference: start at 0.98 and swap our
value-learning components in one at a time (and combined), probing each.

| reference config | fired_auroc | Δ |
|---|---|---|
| baseline (MSE loss, no value target-net) | 0.977 | — |
| + EMA value target-net (τ=1e-3) | 0.934 | −0.04 |
| + Huber value loss | 0.954 | −0.02 |
| **+ Huber AND value target-net** | **0.743** | **−0.23** |

**The two are individually mild but strongly *super-additive*.** Together they drop the reference
from 0.98 to 0.74 — most of the reference→ours gap (our full stack is 0.62; the residual ~0.12 is
the value bound / per-component optimizer / nnx-vs-linen FE, all secondary).

**Mechanism.** The memory is built *incidentally* by the value gradient shaping the recurrent
encoder to predict returns over the sequence (Part E: value-only reference = 0.92). Two of our
choices each throttle that gradient, and compound:
- the **Huber** loss caps the per-sample gradient beyond δ=1 (a soft grad-clip on the value signal);
- the **slow EMA value target-net** (τ=1e-3) makes the regression target laggy/stale.
So the value learns both *weakly* (capped) and *slowly* (laggy target) → the value→encoder gradient
is too feeble to build the board memory. The reference has **neither** (plain MSE to a
current-value bootstrap), so its value gradient is strong and memory forms robustly.

This also explains why our single-factor ablations (Part G: τ=1 alone, bound-off alone) stayed
flat — **you must remove *both* the Huber and the target-net together**; neither alone suffices.

## Synthesis (resolved)

1. On Battleship, spatial board memory is **policy-relevant, not value-relevant** for *one-step*
   prediction (≤1.4% value R², Part D), yet the recurrent value gradient builds it **incidentally**
   under BPTT — the reference forms 0.92 from value alone (Part E).
2. Off-policy is **not** the barrier: staleness, importance weights, update:collect ratio, buffer,
   and LR all match PPO without recovering full memory (Parts A, G). A small effective LR (slow FE)
   *does* unlock partial memory (chance→0.64), confirming carry-amplified drift was the initial
   blocker — but it plateaus.
3. The residual plateau is the **value-learning objective**: our **Huber loss + slow EMA
   value-target-net interact to starve the value→encoder gradient**, which is what forms the memory.
   Fix (both, together): plain **MSE** value loss + **no value target-net** (bootstrap the current
   value), which is exactly the reference's recipe.

**Durable engineering deliverables** (all additive, default-preserving flags): `--grad-clip`,
`--ppo-clip-eps`, `--vtrace-normalize-advantage`, `--scalar-v-bound`,
`--scripted-behaviour`/`--scripted-epsilon`, `--online-batch-size`,
`--collect-steps-per-cycle`/`--updates-per-cycle` (PPO-style collect:update loop), `--rho-bar`/`--c-bar`;
reference-harness switches `FREEZE_POLICY`, `LD_WEIGHT`, `REPLAY_ROLLOUTS`, `RECOMPUTE_TARGETS`,
`USE_VALUE_TARGET_NET`/`VALUE_TARGET_TAU`, `USE_HUBER_VALUE`.

**Fix in our stack did NOT transfer.** Huber→MSE + τ=1 (no target-net) in our PPO-cycle stack
stayed at ~0.56 (100 rounds, trained-and-flat). So the Huber×target-net interaction is a real
handicap *in the reference* but is not our stack's actual ceiling — our stack has a separate,
dominant bottleneck the reference lacks.

## Part I — The FE is not the bottleneck (capacity ceiling)

To localise that residual we (i) verified obs parity (the reference GRU sees only `[hit]`+prev-action
too — the mask is used solely for logit-masking, `models.py:392`), (ii) scoped the optimizer
difference as minor (per-parameter Adam at equal LR; our per-component grad-clip is *gentler* on the
FE than the reference's global clip), and (iii) measured the FE's **supervised** ceiling — train our
exact nnx `paper-arch` GRU to predict the board from the carry, no RL, scaling data:

| supervised episodes | held-out fired_auroc |
|---|---|
| 400 | 0.789 |
| 3,000 | 0.925 |
| 12,000 | **0.965** |

The ceiling **rises with data** (data-limited, not capacity), reaching ~0.965 — essentially the
reference's level. **Our encoder is fully capable.** So the residual is definitively in **how our
off-policy RL objective shapes the encoder**, not the encoder, the observation, the value-loss form,
the target-net, the training rhythm, the buffer, the importance ratios, or the optimizer — every one
of which has now been individually eliminated.

## Bottom line (as of Part I — superseded by Part J)

Off-policy memory formation is **not** blocked by off-policy learning per se (the reference forms it
under 128-deep replay), nor by our encoder (0.965 supervised), nor by any single training-regime
knob. It is unlocked *partially* by taming carry-amplified drift (slow FE: chance→0.64), and the
λ-discrepancy value objective — which only weakly *needs* board memory (≤1.4% value R²) — builds it
only *incidentally*; our stack builds less of it (~0.62) than the reference's value gradient (0.92)
for a residual reason not captured by any component swapped in isolation. The faithful open question
is therefore narrow and specific: **what about our multi-loss off-policy value update makes its
incidental FE-memory gradient weaker than on-policy PPO's**, given identical encoder capacity.

---

## Part J — Forward ladder: build the off-policy agent *out of* the reference

Every part above bisects **backwards** (from our stack, removing suspects) or swaps *single*
our-components into the reference. That design is structurally blind to **interactions**, and it
produced a long list of nulls plus one non-transferring finding (Part H). Part J reverses the
direction: start from the **working reference** and walk cumulatively to our configuration, one
component per rung, probing memory at 4 checkpoints per run.

**Harness** (throwaway, `scratchpad/offpolicy_ladder/`): the reference train loop rewritten as a
single segmented trainer with one switch per rung; encoder and env imported *unchanged* from the
pristine reference package. Gated on 30 numeric checks — `LadderNet` has a byte-identical param
tree and identical outputs to `BattleShipActorCriticRNN`; GAE matches an independent numpy loop;
V-trace with ρ≡1 matches GAE (<1e-3); V-trace with real ρ matches numpy; sampled buffer windows are
time-contiguous, never straddle the write pointer, and episode-aligned starts satisfy `done[t-1]`;
`LadderBattleship(bonus=100)` reproduces the reference reward trace exactly; recomputed `log π(a)`
equals stored `log μ(a)` to 0.0 when params have not moved. 47 runs × 2e6 env steps.

**Parity gate.** R0 (untouched reference) = **0.975 / 0.996** over two seeds, matching the
historical 0.977–0.999. Band ≈ [0.97, 1.00]; treat < ~0.93 as a break.

| rung | cumulative swap (keeps all previous) | ε=1 fired_auroc |
|---|---|---|
| R0 | reference LD-PPO, 2 seeds | **0.975 / 0.996** |
| R1 | + V-trace λ-return recursion (ρ≡1) | 0.998 |
| R2 | + real IS ratios ρ=min(1,π/μ), recomputed targets, 2 seeds | 0.962 / 0.961 |
| R4b | + flat FIFO window buffer, episode-aligned, zero carry, 1st-episode mask | 0.977 |
| R5a_iso | + update:collect ratio 16 (reuse 4→16) | 1.000 |
| R6f | + EMA value target-net τ=1e-3 | 0.972 |
| R7f | + env parity: terminal bonus 0, γ=0.99, single env | 0.973 |
| R8f | + Huber value loss (no PPO value-clip) | 0.993 |
| R9f | + sigmoid value bound V=−100·σ(raw) | 0.938 |
| R10f | + separate fe/actor/critic Adams, eps 1e-8, no anneal | 0.961 |
| R11f | + separate Huber λ-discrepancy loss with its own coefficient | 0.969 |
| **R12f** | **+ our V-trace PG actor** (Â=ρ(r+γv′−V), λ=1 trace, ppo-clip, advnorm) | **0.865** |
| **R13af** | **+ per-action Q critic**, value loss on Q(s,aₜ) only | **0.722** |
| **R13bf** | **+ twin-Q, min over twins (SAC-style V)** | **0.708** |

**Every rung through R11 stays at or near reference level.** The entire collapse is the last two
rungs, and the terminal rung (0.708) reproduces our stack's failure regime (0.55–0.64).

### The answer: sparse per-action regression × diffuse actor

| run | ε=1 |
|---|---|
| our V-trace actor on R0 **alone** | **0.983** |
| per-action Q **without** our actor | **0.988** |
| both (R13af) | **0.722** |
| both + **dense value loss** (regress V=Σₐπ(a)Q against the same target) | **0.977** |
| both + twin-Q + dense value loss | **0.969** |
| both + lower entropy coef instead (entropy 3.43→2.80) | 0.878 |
| R12f without the ppo-clip trust region | 0.591 (entropy → 5e-4) |

Both components are **benign in isolation and fatal together**. Mechanism: a per-action Q head
trained only on `Q(s,aₜ)` updates **1 of 100 output columns per step**, so the gradient reaching the
shared recurrent encoder is ~100× sparser than a scalar-V head's, and *which* column receives it is
dictated by the behaviour policy's samples. Our actor holds the policy near-uniform (entropy 3.4 of
ln 100 = 4.6), scattering those updates into noise; the PPO actor concentrates it (~2.0) and the same
sparse loss is harmless. Making the loss **dense** recovers +0.26 with no change to the actor;
concentrating the policy instead recovers only +0.16, so the loss structure dominates and the
diffuseness is secondary. Separately, the ppo-clip trust region is load-bearing: without it the
policy collapses to deterministic within a few steps and memory dies.

### Three earlier conclusions corrected

1. **Off-policy trap found in the replay *ring*, and the V-trace caps DO matter** (contra "caps
   hypothesis NEGATIVE"). Ring N=8/32/128 → 0.951 / 0.995 / **0.736**; replay-128 with GAE alone
   0.938; with ρ≡1 + recomputed targets 0.998; with **ρ̄=c̄=5 → 0.994**; with the ring slot
   **resampled per update → 0.997**. The one-sided `ρ=min(1,π/μ)` truncates the upside but not the
   downside, so E[ρ]<1 on stale data and the reward correction is attenuated — but only when the
   cycle's updates hammer *one* stale rollout. The flat window buffer never suffers it
   (R4a 0.999, R4b 0.977, stored-carry R4c 0.994), so this is not on the path to our stack.
2. **Memory formation is DATA-limited; gradient reuse is a true null.** 2×2: reuse 4 → 0.787 (500k
   steps) / 0.975 (2e6); reuse 16 → 0.793 (500k) / **1.000** (2e6). This corroborates the
   PPO-cycle and batch-size nulls of Parts A/G with a positive explanation, and mirrors the
   supervised data-scaling curve of Part I (0.79@400ep → 0.925@3k → 0.965@12k).
3. **The +100 terminal bonus buys sample-efficiency, not memory.** At a 125k-step budget dropping it
   costs 0.10 (0.865 vs 0.968); at 2e6 steps it costs nothing (0.973 vs 0.972). γ=0.99 and
   single-env collection are free at either budget.

### Vindicated: our stabilizers are load-bearing off-policy

The knobs Part H found *harmful in the reference* are helpful or neutral here — the EMA value
target-net rescues the high-reuse regime (0.870 → 0.968 at reuse 64), Huber is neutral-to-positive
(0.993), the sigmoid bound and the separate optimizers cost nothing (0.938 / 0.961), and recasting
the λ-discrepancy as its own Huber loss costs nothing (0.969). Same knobs, opposite sign, because
off-policy bootstrapped targets are divergence-prone where on-policy GAE is not.

## Bottom line (resolved by Part J)

Off-policy λ-discrepancy memory formation is **not** blocked by off-policy learning, replay,
staleness, importance weighting, window sampling, carry handling, gradient reuse, target-nets, loss
form, value bounds, optimizers, γ, reward shape, or encoder capacity — a cumulative ladder passes
all of them at reference level (0.94–1.00). It is blocked by the **critic's output structure
interacting with the actor's entropy**: a per-action (twin-)Q head regressed only at the executed
action starves the shared recurrent encoder of gradient, and our near-uniform off-policy actor
maximises that starvation. The fix is to make the value loss dense over actions (regress
`V(s)=Σₐπ(a|s)Q(s,a)` against the same target, i.e. an expected-SARSA-style projection) — which
restores 0.708 → 0.969 without touching anything else — and to keep the ppo-clip trust region.

**For `iqlearn.py`:** add the dense/expected value-loss term alongside (or instead of) the
taken-action twin-Q regression, keep `--ppo-clip-eps`, and stop treating the reward/γ/buffer/timing
knobs as suspects. The earlier ~0.62 plateau is consistent with this diagnosis: every stabilizer we
added was helping, while the SAC-style per-action critic was quietly starving the encoder.

---

## Part K — Porting the fix: the probe FLOOR, and active erasure

Part J's fix was implemented in `iqlearn.py` (branch `dense-value-loss`, five sparse
sites incl. the always-active `loss_critic`) with the point-mass identity and
gradient-reach properties unit-tested. The first six 2e6-step runs came back flat at
0.53–0.57. The diagnosis that followed changed the question.

### K.1 The probe floor — and what it does to the project's history

**An UNTRAINED agent scores 0.599 / 0.599 / 0.621** (3 seeds, 200 training steps),
and an agent whose encoder is frozen at random init (`--fe-lr 0`) holds **0.580–0.588
flat** over 5 rounds. A randomly-initialised GRU echoes recent hit/miss history, so
the probe's chance level for this task is **~0.58–0.62, not 0.5**.

⇒ **The entire historical "0.55–0.64 plateau" (Parts A–I) lies at or below the
untrained floor.** The slow-FE recipe's 0.64 — the previous arc's best result and its
one apparent success — is ~0.03 above a random encoder. On this evidence **our stack
has never formed board memory at all**, and every earlier comparison between 0.55 and
0.64 was noise around chance. The floor was never measured until now; it should
accompany every future number.

### K.2 λ-discrepancy weight — not the LR — is what diverges the value

Stabilisation sweep (300k steps, dense fix on, probe vs floor):

| arm | `v` | `critic_loss` | `ld_loss` | verdict |
|---|---|---|---|---|
| λ-coef 0.1, normal LR | −27.8 → −37.4 | 0.76 → **0.23** | 3.9 → 0.15 | stable |
| λ-coef 0.1 + LR 1e-5 | −20.9 → −37.3 | 3.1 → 0.61 | 4.4 → 0.12 | stable |
| LR 1e-5, λ-coef 1.0 | −24.6 → −43.8 | 10.8 → 2.2 | 6.6 → 0.93 | stable |
| τ 0.005 → 1e-3 | −9.7 → −46 | 0.78 → **7.5** ↑ | 25 → **57** ↑ | diverges |
| control (λ 1.0, normal LR) | −38.5 → **−95.7** | 9.2 → 10.5 | 6.7 → **28.6** ↑ | diverges |

At γ=0.99 with −1/step the value is bounded by −100; the control reached **−116**
earlier, i.e. past the physical fixed point. Dropping λ-coef 1.0 → 0.1 at *normal* LR
gives the lowest critic loss of the sweep (0.23 vs 10.5). Slower τ makes it worse.
⇒ the λ-discrepancy *pressure* destabilises the value, inverting Parts C/H's framing
(value divergence as the primary fault, discrepancy explosion as its symptom).
**The dense fix also demonstrably stabilises the value**: −76 worst-case and inside
the bound, vs −116 divergence for the matched sparse control.

### K.3 Training ERASES the encoder's board information

With the value stable (`critic_loss` 0.02–0.30), the probe is **flat at 0.53 across
all runs — BELOW the 0.58 frozen-encoder floor**, and it is already there at round 1
(20k updates), not by slow decay:

| | probe curve (probe every round) |
|---|---|
| FE frozen at random init | 0.587 0.582 0.580 0.583 0.588 |
| FE trained (stable value + dense fix) | 0.535 0.536 0.529 0.536 0.530 |

⇒ the failure is not "our objective fails to add memory" but "**our objective
actively destroys the echo a random GRU has for free**", within the first 20k
updates, and then sits at the degraded level. Consistent with a state-independent
minimiser: the cheapest descent direction suppresses the input pathway.

### K.4 BPTT is sound (ruled out)

`scratchpad/bptt_diag.py`, three checks on the same `calculate_latent` path training
uses: (a) forward — `obs[0]` reaches `latent[-1]` (Δ 4.3e-3 from a Δ5 perturbation);
(b) gradient — `d latent[-1]/d obs[0]` = 2.3e-2 (vs 1.07 at the last step);
(c) the GRU hidden→hidden kernel receives gradient 0.82 from a last-step-only loss,
which is only creditable through time. Temporal credit flows. NB the signal decays
~275× (forward) / ~45× (gradient) over 11 steps, so retention over Battleship's ~90
steps must be *learned* — it is not free at init.

### K.5 Status

Ruled out this session: probe artefact (frozen-FE control is flat), value instability
(fixed, memory unchanged), BPTT plumbing, slow-LR recipes (5 flat probes over 560k
steps), τ. Established: the floor, the λ-pressure/value-divergence inversion, active
erasure, and that the dense fix helps stability but not memory *in our stack* (where
it recovered 0.708 → 0.969 in the ladder harness). The open question is now sharp:
**what does the reference's training loop do that makes memory form incidentally
under any objective, when ours erases it under all of them** — given identical obs,
capacity, BPTT, and stability.

### K.6 The weekend sweep: no objective changes anything

Every arm below ran the *stabilised* configuration (λ-coef 0.1 → `critic_loss` 0.02–0.6,
`v` well inside its bound). Floor to beat: **0.58–0.62**.

| arm | what it tests | probe |
|---|---|---|
| dense on / off, 2e6 steps | the Part-J fix in our stack | 0.52–0.54 |
| slow LR 1e-5 (all / FE-only) | the 32× optimisation-pressure gap | 0.53 (5 flat probes) |
| τ 1e-3 | target-net lag | diverges |
| value-only + **frozen random policy** | the reference's minimal memory-forming regime (it gets 0.92 here) | 0.52 |
| + **γ=1.0** | the reference's discount | 0.52 |
| + **γ=1.0 and the +100 reference reward** | the reference's exact reward | 0.53 (flat to r46) |
| `--sac-critic-coef 0` (drop the 1-step Bellman head) | the memoryless-satisfiable always-on target | 0.53 |
| λ pair 0.95/0.99 (both near-MC) | short-λ bootstrapping | 0.55 |
| **GVD, spatial cumulant** | a target that is board-shaped *by construction* | 0.53 |
| GVD, scalar cumulant | GVD without the spatial target | 0.53 |
| no `--paper-arch` (plain GRU) | the encoder implementation | **0.513–0.533, 19 probes to 1.94M steps** |

### K.7 Controls: the comparison is valid

- **Probe reads on-distribution carries.** `collect_rollout` does
  `carry_out = where(done, zero_carry(), new_carry)` — episode-reset exactly as
  episode-aligned training windows are, recording the within-episode carry.
- **Probe is not drifting**: frozen-encoder control flat at 0.580–0.588 over 5 rounds.
- **Encoder is capable**: 0.925 supervised on the same FE and the same metric (Part I).
- **Time alignment correct**: perturbing `a_k` first moves `latent[k+1]`, `o_k` moves
  `latent[k]` — step *t* sees `(o_t, a_{t-1})`, so hits are attributable to cells.
- **BPTT sound**, **latent routing** un-detached, **buffer contiguity** covered by tests.
- **The ladder target reproduces**: R13b+dense at a fresh seed → 0.799 → 0.825 → 0.904
  → **0.953** (originally 0.969). Not a lucky seed.

### K.8 Verdict and recommendation

| same probe, same metric | result |
|---|---|
| our nnx FE, supervised board loss | **0.925** |
| reference loop hosting OUR components (V-trace targets, our actor, 128-deep replay, dense per-action twin-Q) | **0.953–0.969** |
| our loop, **every** objective tried incl. a board-shaped one | **0.51–0.55** (floor 0.58–0.62) |

The encoder is capable; our components demonstrably work when hosted in the reference's
training loop; our loop fails under every objective — including one whose target is
board-shaped by construction — and lands *below* what an untrained GRU carries for free.
⇒ **The defect is in our training loop, not in the objective, the encoder, the value
machinery, or any component the forward ladder swapped.** No further objective-level
knob is worth testing.

**Recommendation.** Stop repairing `iqlearn.py`'s loop. Build off-policy imitation on the
reference's training loop, which the ladder has already shown will host our pieces at
0.95+: V-trace λ-return targets, real IS ratios, 128-deep replay, an episode-aligned
window buffer, the EMA target-net, and the per-action twin-Q critic **with the dense
value loss** (which is what lifts that configuration from 0.708 to 0.969, and which also
demonstrably bounds the value where the sparse form diverges past −100). The ladder
harness in the session scratchpad is the working starting point; the remaining work is
productionising it rather than diagnosing it.

**Unfinished thread**: *which* aspect of our loop erases memory is still unidentified —
BPTT, alignment, routing, buffer, windows, capacity and probe are all individually
verified sound, so it is likely an interaction among them. Answering it is optional for
the project goal (the recommendation above routes around it) but it is the one honest
loose end.

### K.9 Pre-port bug audit (asked for before committing to a port)

The signature — below-floor, immediate, objective-independent, while every component
verifies — is more typical of a subtle bug than of a deep truth, so the data path was
audited end-to-end rather than by component:

| check | method | result |
|---|---|---|
| probe decodes the state the loss shapes | `CARRY_DIM == memory_hidden_dim == 512`; for a GRU the cell output **is** the carry, so `latent == carry` | ✔ same array |
| training samples the right buffer | `update_step` passes `sac.online_buffer`, not the dummy 1-transition expert buffer the probe builds from zeros | ✔ correct |
| (obs, action) pairing in the buffer | `run_env_step` stores `(o_t, a_t)`; `calculate_latent` shifts so step *t* sees `a_{t-1}` (numerically verified) | ✔ correct |
| **trajectory coherence of the real stored data** | Battleship invariant on 4905 stored transitions: `mask[t] == mask[t-1] \ {a_{t-1}}`, one cell consumed per step, `reward == -1` off-terminal | ✔ **all PASS** |
| episodes fit the training window | mean 91.5, max 100 vs window 120 and kept region [0:100] | ✔ 0/53 overflow |
| hit-bit statistics | 0.140 vs ~0.15 expected (14 ship cells / ~92 shots) | ✔ plausible |
| **encoder memory horizon at init** | flip the t=0 hit bit, track \|Δlatent[t]\|: **ours half-life 2 steps, reference half-life 2 steps**, ratio 1.1× at t=90 | ✔ no architectural handicap |

No defect found in the data, the pairing, the buffer selection, the probe target, the
BPTT path, the routing, the capacity, or the encoder's initial dynamics. **Both encoders
forget within ~5 steps at initialisation — retention over ~90 steps is LEARNED, and the
reference's loop learns it while ours does not.**

**Correction to K.3.** "Training actively erases the echo" over-reads the numbers.
Trained 0.53 vs frozen 0.58 is a 0.05 gap against a floor that itself spans 0.58–0.62
across seeds, so the defensible claim is the weaker one: **training leaves the encoder at
or near the untrained floor — no board memory forms — and any erasure is marginal.**
The conclusions in K.5/K.8 do not depend on the stronger claim.

**The one test not yet run** (deferred as too invasive to add unsupervised): an auxiliary
memory loss inside our loop — predict the within-window accumulated hit-map from the
carry, a target computable from stored actions+hit-bits alone. If our loop reaches ~0.9
under it, the loop is sound and the entire story is that no RL objective here demands
retention; if it stays ~0.55, the loop cannot shape the encoder and porting is the only
option. That is the highest-value first move if certainty is wanted before porting.
