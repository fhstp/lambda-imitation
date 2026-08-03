# Our stack vs the forward-ladder harness — component-by-component differences

**Why this document exists.** The forward ladder (`ablations.md` Part J) built an
off-policy agent *out of* the working on-policy reference and reached order-invariant
`fired_auroc` **0.953–0.969** while hosting our V-trace targets, our actor, 128-deep
replay and a dense per-action twin-Q critic. Our own stack, on the same task with the
same probe and metric, sits at **0.51–0.55** against an untrained-encoder floor of
**0.58–0.62** — i.e. no board memory at all (`ablations.md` Part K). Since the ladder
proves our *components* work when hosted in the reference's loop, the cause must be
something present in our stack that no ladder rung ever contained. This is the
systematic inventory of exactly that.

Three parallel audits covered the data path, the losses/value machinery, and the
model/optimiser. Every claim below that affects a conclusion was re-verified directly;
where an audit described a **default** that our experiments had overridden, that is
called out rather than repeated.

---

## 1. Findings that invalidate earlier results

### 1.1 `--gvd` defaults to **True** — every weekend arm silently trained a GVD branch

`examples/lambda-envs/battleship_board_probe.py:157` — `parser.set_defaults(gvd=True)`.

Consequently **every** arm run over 2026-08-01/02 (the stabilisation sweep, the
"value-only" arms, the reference-matching arms, the `--sac-critic-coef 0` arms, the
loop-validation arms, the aux diagnostic) carried the successor-feature branch into the
shared encoder: two `Head(512→256→256→100·n_features)` heads (~224k params each), their
own Adam optimisers at `gvd_sf_lr=1.8e-4`, their own EMA targets, two sparse per-action
V-trace TD losses, and the reward-free SF discrepancy at `gvd_coef=0.2`. Earlier sessions
had already recorded these heads diverging and eroding memory
([[battleship-10x10-scaling]]).

What this breaks:

- the "**value-only**" arms (`--lambda-coef 0`) were never value-only;
- the "**drop the 1-step critic**" arms (`--sac-critic-coef 0`) still had SF losses
  shaping the encoder;
- `loopval_gvdscalar` was **configuration-identical** to the arms being used as controls
  — which is why every arm reported an identical ~0.53;
- the "reference-matching" arms were not reference-matching: the reference has no SF
  branch at all.

Every log printed `Building SAC+LD+GVD agent …` on its first line.

### 1.2 The V-trace tail bootstrap asserts `V(s_{T+1}) = 0` — verified, and it bites at λ≥0.95

Three sites initialise the backward recursion with `(v_{T'}, V_{T'}) = (0, 0)` in
**absolute** value form: `iqlearn.py:2569` (λ-critics), `iqlearn.py:1962` (V-trace actor),
`iqlearn.py:698` (`sf_vtrace_targets`, GVD). The docstring at `iqlearn.py:661-663`
documents this as intentional — "the missing bootstrap mass at the unroll tail is
absorbed by the caller dropping the last `lambda_truncation` steps" — but the magnitude
was never checked. It decays only as `(γλ)^trunc`, and with `lambda_truncation=20`
(hardcoded, `battleship_board_probe.py:1208`):

| λ | true target | our last *kept* target | error | surviving bias `(γλ)²⁰` |
|---|---|---|---|---|
| 0.75 | −100.0 | −99.7 | +0.3 | 0.003 |
| **0.95** | −100.0 | **−71.0** | **+29.0** | 0.29 |
| 0.99 | −100.0 | −33.8 | +66.2 | 0.67 |
| **1.0** | −100.0 | **−19.0** | **+81.0** | **0.82** |

(Steady state `V = −1/(1−γ) = −100`, reward −1/step, γ=0.99, no terminations, ratios 1.
The ladder is exact at every λ: it carries `(v − V)` so its zero init asserts nothing,
and it stores `obs_next`/`done_next` per slot for a real bootstrap — `ladder.py:399-417`,
`427-428`, `611-618`.)

Consequences, all of which hit configurations we have actually run:

1. **Every weekend run used `--lambda2 0.95`**, so the long-horizon head — the one whose
   target encodes steps-remaining and therefore *requires* integrating past hits — was
   regressing targets biased ~29 units toward zero.
2. **`loss_actor_vtrace` hardcodes λ=1** (`iqlearn.py:1949`), where **82 %** of the bias
   survives truncation, so every advantage was biased positive across the whole kept
   window. This retro-explains the "large and drifting `adv_raw_mean`" that
   `--vtrace-center-advantage` and `--vtrace-normalize-advantage` were added to
   compensate for: those were band-aids over this bug.
3. The `λ = 0.95/0.99` ablation arm (`ablations.md` K.6) was regressing largely garbage.
4. Porting the reference's λ=(0.1, **0.95**) pair into our stack would have been silently
   broken.

### 1.3 `--sac-critic-coef 0` was a confounded ablation

With that coefficient at 0 the main twin-Q receives **no** gradient (the actor's use of it
is `stop_gradient`'d), so it remains at random initialisation while still serving as the
V baseline for `loss_actor_vtrace`. The arm therefore did not "remove the 1-step
pressure"; it replaced it with a randomly-initialised baseline.

---

## 2. The gradient-budget inversion (measured, not argued)

`‖∂term/∂FE‖` (optax global norm over the encoder subtree only) for every live loss term,
at Battleship-10×10 shapes with real head dims:

| encoder-gradient share | ours | ladder terminal rung |
|---|---|---|
| **λ-discrepancy** (the memory mechanism) | **6–7 %** | **70 %** |
| 1-step MSE Bellman critic (sparse + dense) | 37–41 % | — (no counterpart) |
| GVD successor features (3 terms) | 17–19 % | — (no counterpart) |
| `E_π[Q]` ascent through the encoder | 9–11 % | — (no counterpart) |
| dense λ-return regression | 22–26 % (mostly the *sparse* form) | 30 % |
| **live terms shaping the encoder** | **13** (5 with probe defaults) | **2** |

The two systems are close to inverted. The ladder's encoder is shaped almost entirely by
the λ-discrepancy plus a dense λ-return regression — both Huber, both dense, both on the
same two heads, one coherent objective. Ours gives the discrepancy a **6 %** voice among
terms that are satisfiable *without* memory.

A second measurement explains why the sparse terms are worse than their share suggests.
Gradient **coherence** on the latent (`‖mean_s ∂L/∂z‖ / mean_s‖∂L/∂z‖`):

| | coherence |
|---|---|
| our five sparse (executed-action) terms | **0.039–0.055** |
| dense terms (ours and the ladder's) | 0.197–0.327 |
| ladder value / ld | 0.308 / 0.505 |

Our sparse terms carry 4–7× the per-sample magnitude of the dense ones for a comparable
*coherent* component: they dominate Adam's second-moment estimate while contributing an
incoherent direction. That is the Part-J mechanism, now present five times over
(λ1, λ2, SF1, SF2, main critic) and crowding out the one term that does the memory work
in the system that succeeds.

---

## 3. Structural differences with no ladder counterpart

| # | difference | detail | why it plausibly matters |
|---|---|---|---|
| 1 | **Per-transition gradient reuse 15,360 vs 4** | batch 128 (hardcoded `probe:1199`) × 120-step windows × 1 update/env-step. Max reuse ever tested anywhere: **64**. The "PPO-cycle proves timing isn't the blocker" test ran at reuse ≈ **300** — still 75× the ladder's winner. | 3,840× outside the tested range; an encoder can fit a fixed 200k ring rather than learn to integrate online |
| 2 | **`loss_ld` freezes both heads** (`iqlearn.py:2200-2201`) | 100 % of the discrepancy gradient lands on the encoder; the ladder lets the heads absorb **42 %** | with no escape valve, the cheapest way to make two frozen heads agree is to move the latent into their (state-independent) agreement region |
| 3 | **Three value systems on one encoder** | main 1-step twin-Q + λ1 + λ2 (6 Q nets) + 2 SF heads = **8 heads**; the λ-critics feed nothing but their own TD loss and `loss_ld` | the ladder has one value system that is *simultaneously* the discrepancy pair and the actor's baseline |
| 4 | **1-step Bellman target in MSE** | `iqlearn.py:2118`; always on; untruncated | `V ≈ r + γV(s')` is satisfiable by a near-constant, history-free value; MSE gradient grows linearly with the residual |
| 5 | **LayerNorm inside every value head** | `critic_layer_norm=True` (`probe:124`); no rung has LayerNorm anywhere | strips latent *scale* before the value read-out, so magnitude-coded accumulation over 90 steps cannot be rewarded. NB our own earlier datapoint said LN *helped* (0.61 vs 0.55) — contradictory, worth an explicit test |
| 6 | **λ-critic heads `(64,64,64)`** | hardcoded `probe:1295-1296`, vs the ladder's single 512-wide head per λ | an 8× narrower head can satisfy both λ targets from a low-rank projection of the carry, which need not encode the board |
| 7 | **Sparse *plus* dense** value loss | `sparse_value_loss=True` by default alongside `dense_value_coef` | the ladder's 0.708→0.969 fix was a **replacement**; keeping the sparse term retains the gradient the fix was meant to remove |
| 8 | **λ-discrepancy at the executed action** | `dense_discrepancy=False` by default, and never run | the ladder's ld is always on `V = Σπ(a)Q` |
| 9 | **`min(q1,q2)` as the regression prediction** | `get_q`, `iqlearn.py:1380` | gates gradient to one twin per sample and biases values down; every rung regresses each twin independently |
| 10 | **EMA τ = 0.005 per update** | vs the ladder's 1e-3 per 16 updates ⇒ ~6e-5/update | our target tracks ~80× faster (2560× in env-step terms). NB `ablations.md` K.2 found τ=1e-3 *diverges* in our loop, so the ladder's own setting is currently unreachable here |
| 11 | **`critic_lr` 2e-4 vs `fe_lr`/`actor_lr` 1e-4** | `probe:99-101` | a 2× faster value head on a shared encoder; the ladder never ran unequal LRs |
| 12 | **Two encoder-facing losses ignore the step mask** | the 1-step critic and the V-trace actor; the ladder masks **every** term (`ladder.py:646-649`) | cross-episode-boundary steps train the encoder even with `--mask-first-episode-only` set |
| 13 | **`lambda_truncation=20` drops 27 % of steps** | and only from the λ/SF/actor losses, not the critic or the discrepancies | the *set* of terms shaping the encoder differs by timestep; unnecessary in the ladder, which bootstraps exactly |
| 14 | **No value bound** | the ladder's terminal rung has `V = −100·σ(raw)` (R9) | our value has been observed at −116, past the physical fixed point |
| 15 | **nnx GRUCell fuses the recurrent kernels** | one `orthogonal()` over (512,1536) ⇒ each gate slice gets **1/√3 ≈ 0.578** of unit norm, vs linen's three separate (512,512) at 1.0; also missing `b_hn` (−512 params) | recurrent path starts 0.577× weaker. Measured half-lives are equal (~2 steps) so this is *not* the cause, but it is a genuine library-level mismatch |

---

## 4. Verified equivalent — not the cause

- **Encoder graph up to and including the GRU input is byte-for-byte the ladder's**: same
  widths, same `orthogonal(√2)`, same hit skip, same 101-wide `[hit ⧺ onehot(a_{t-1})]`
  input, and `latent == carry` on both sides.
- **Encoder capacity and initial dynamics**: 0.925 supervised on this same FE and metric;
  memory half-life at init ~2 steps for *both* encoders (ratio 1.1× at t=90).
- **Time alignment**: perturbing `a_k` first moves `latent[k+1]`, `o_k` moves `latent[k]`.
- **Trajectory coherence of the stored data**: Battleship invariants pass over 4905
  transitions (`mask[t] == mask[t-1] \ {a_{t-1}}`, one cell consumed per step, rewards).
- **BPTT**: forward and gradient reach t=0; the recurrent kernel receives credit from a
  last-step-only loss.
- **Buffer**: contiguity and wraparound covered by `tests/test_buffer.py`; `update_step`
  passes `sac.online_buffer`, not the zero-filled expert buffer.
- **Probe**: reads episode-reset, on-distribution carries; frozen-encoder control flat at
  0.58; our probe's target is marginally *harsher* than the ladder's.
- **Env / reward / γ / env-count**: exact at `--terminal-bonus 0`.
- **Adam eps and no-LR-anneal**: match ladder R10+ exactly.

Also **not applicable** to our runs (audits described defaults we had overridden):
missing grad clipping, missing PPO clip, the SAC `E_π[Q]` actor, λ-critics off,
`fake_onpolicy_loss=True`, and stale stored carries (episode-aligned windows start where
the stored carry is already zero).

---

## 5. Ranked actions

1. **Fix the tail bootstrap** at all three sites — carry `(v − V)` so the zero init
   asserts nothing. Cheap, and no result measured with λ=0.95 or the λ=1 actor is
   trustworthy until it is done.
2. **Make the defaults ladder-matched**, and print an explicit divergence banner at
   startup, so a silent `gvd=True`-class error cannot recur. (§6)
3. **Re-test with GVD off** — the arm that should have come first.
4. **Rebalance the encoder gradient budget** toward the discrepancy: unfreeze the ld
   heads, use the dense discrepancy, replace rather than add the sparse value term, and
   consider dropping the 1-step critic (properly this time, i.e. without leaving a random
   baseline in the actor's path).
5. **Reach ladder-matched reuse (~4)** — needs both a batch knob and a collect/update
   cycle; both already exist on `archive/offpolicy-ladder-knobs`.
6. **Single-factor tests** for LayerNorm, λ-head width, `critic_lr`, and the fused-GRU
   recurrent scale.

Only after 1–4 have been tried does "port onto the reference loop" become the
evidence-backed recommendation; the case for it in `ablations.md` K.8 rested on runs that
§1.1 and §1.2 have now compromised.
