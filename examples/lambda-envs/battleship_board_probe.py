"""Board probe for Battleship: train a probe (memory -> ship occupancy) and visualise.

Battleship analogue of ``pocman_pellet_probe.py``.  The agent only ever observes
a single bit per step — did the last shot hit — so to localise ships at all its
recurrent memory must integrate the *action–observation history*.  The previous
action therefore reaches the feature extractor via ``use_prev_action=True``
(packed-carry prev-action input; the legal-action mask stays out of the
network, used only to mask the policy).  A probe then tests whether the memory
has reconstructed the hidden board.

Pipeline:
  1. Train a Battleship SAC(+LD) agent (action-masked)
  2. Collect rollouts (auto-reset) saving hidden states + the true ship board
     and the hits/misses fired so far
  3. Train a 3-layer MLP probe: hidden state -> ship occupancy (BCE)
  4. Visualise probe predictions vs the true board, with the agent's shots
     overlaid, on the rows×cols grid

All artefacts are saved under --output-dir so individual phases can be skipped
on subsequent runs (--skip-train, --skip-collect, --skip-probe).  Vis-only mode
(--vis-only) needs only dataset.pkl + probe.pkl (just jax + matplotlib).

Usage:
    python battleship_board_probe.py                          # full pipeline
    python battleship_board_probe.py --rows 5 --cols 5        # smaller board
    python battleship_board_probe.py --skip-train             # reuse saved agent
    python battleship_board_probe.py --no-approximate-lambda  # SAC-only baseline
    python battleship_board_probe.py --vis-only --mp4         # just render (portable)
    python battleship_board_probe.py --wandb                  # log to W&B
    python battleship_board_probe.py --probe-eval-interval 0  # disable periodic probe eval

With --wandb the agent-training metrics, probe-training loss, and final probe
accuracy/visualisations are logged.  By default a lightweight probe is also
trained and scored every --probe-eval-interval rounds during agent training
(logged under ``probe_eval/``), tracing how board decodability grows over time.
"""

import argparse
import os
import pickle
import sys
from functools import partial

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Battleship board probe visualisation.")

g = parser.add_argument_group("environment")
g.add_argument("--rows", type=int, default=10, help="board rows (default 10)")
g.add_argument("--cols", type=int, default=10, help="board cols (default 10)")
g.add_argument("--dense-reward", dest="dense_reward", action="store_true",
               help="reward every hit (default: sparse terminal reward)")
parser.set_defaults(dense_reward=False)
g.add_argument("--terminal-bonus", dest="terminal_bonus", type=float, default=None,
               help="sparse-mode reward on the clearing step (default None = "
                    "env default rows*cols; 0.0 = pure -1/step, which is what "
                    "surfaces the spatial-Q spike).")
g.add_argument("--ship-lengths", default="5,4,3,2", metavar="L1,L2,…",
               help="comma-separated ship lengths (default 5,4,3,2).  Use "
                    "shorter ships for small-board curriculum stages, e.g. "
                    "'3,2' on a 5x5 board.")
g.add_argument("--full-obs", action="store_true",
               help="DIAGNOSTIC: append the full per-cell hits_misses grid to "
                    "the observation so the FE sees Markov state (no memory "
                    "needed).  Tests whether the SAC+GVD policy machinery can "
                    "learn good play *given* the board — isolating the failure "
                    "to the memory-incentive gap.")

g = parser.add_argument_group("agent training")
g.add_argument("--rounds", type=int, default=100, help="training rounds (default 100)")
g.add_argument("--train-steps", type=int, default=10_000, help="env steps per round (default 10 000)")
g.add_argument("--seed", type=int, default=42)
g.add_argument("--num-seeds", type=int, default=1,
               help="number of seeds to run; seeds are args.seed + i.  >1 runs "
                    "the full pipeline per seed concurrently (vmapped) and logs "
                    "per-seed (seed_i/) plus aggregated (agg/ mean,std,sterr) "
                    "metrics (default 1; sweep default 5).")
g.add_argument("--concurrent-seeds", type=int, default=0,
               help="seeds trained concurrently in one vmapped+jitted kernel "
                    "(0 = all --num-seeds in a single group, the common case).  "
                    "When set, --num-seeds must be divisible by it.")
g.add_argument("--final-return-window", type=int, default=10,
               help="number of final rounds averaged for the smoothed final "
                    "return / steps-to-clear (default 10).")
g.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
g.add_argument("--memory-hidden-dim", type=int, default=512)
g.add_argument("--projection-dim", type=int, default=128)
g.add_argument("--paper-arch", dest="paper_arch", action="store_true",
               help="use the original lambda-discrepancy Battleship network "
                    "(BattleShipActorCriticRNN): a Dense(2H)->relu->concat(hit)"
                    "->Dense(H)->relu pre-RNN embedding with a hit-bit skip "
                    "connection (H = --memory-hidden-dim), a GRU memory, and "
                    "single-hidden-layer actor/critic heads of width H. "
                    "Overrides --projection-dim and the head dims; the "
                    "prev-action one-hot is still fed via use_prev_action.")
parser.set_defaults(paper_arch=False)
# Value-stability knobs (previously hard-coded in Hyperparameters).  Defaults
# reproduce the prior behaviour exactly so existing launches are unchanged.
g.add_argument("--fe-lr", type=float, default=1e-4, help="feature-extractor lr (default 1e-4)")
g.add_argument("--actor-lr", type=float, default=1e-4, help="actor lr (default 1e-4)")
g.add_argument("--critic-lr", type=float, default=2e-4, help="critic lr (default 2e-4)")
g.add_argument("--alpha", type=float, default=0.1, help="entropy temperature (default 0.1)")
g.add_argument("--autotune-alpha", action="store_true",
               help="auto-adjust alpha to match --target-entropy (SAC discrete; "
                    "default off — alpha held fixed at --alpha).")
g.add_argument("--target-entropy", type=float, default=0.0,
               help="target policy entropy for alpha autotuning (default 0.0). "
                    "For discrete a common heuristic is ~0.5–0.98·ln(num_actions).")
g.add_argument("--gamma", type=float, default=0.99, help="discount factor (default 0.99)")
g.add_argument("--tau", type=float, default=0.005, help="target-net EMA coefficient (default 0.005)")
g.add_argument("--grad-clip", type=float, default=0.0,
               help="global grad-norm clip for FE/actor/critic/SF (0 = off, default). "
                    "Recommended for the recurrent BPTT unroll, e.g. 1.0–10.0.")
g.add_argument("--cql-coef", type=float, default=0.0,
               help="CQL-style action-discrimination penalty on the critic "
                    "(logsumexp_a Q − Q_taken over legal actions), scaled by this "
                    "coef (0 = off, default).  Counteracts a critic that collapses "
                    "to an action-independent (flat) Q; try e.g. 0.5–5.0.")
g.add_argument("--critic-layer-norm", dest="critic_layer_norm", action="store_true",
               help="use LayerNorm in the critic MLP (default on).")
g.add_argument("--no-critic-layer-norm", dest="critic_layer_norm", action="store_false",
               help="disable critic LayerNorm (control: LN can wash out small "
                    "per-action Q differences).")
parser.set_defaults(critic_layer_norm=True)
g.add_argument("--behaviour-epsilon", type=float, default=0.0,
               help="fraction of online-collection steps driven by a scripted "
                    "hunt/target behaviour policy instead of the actor (0 = off, "
                    "default).  Generates board-exploiting trajectories so the "
                    "value function depends on the hidden board — pressuring the "
                    "memory to encode it.  The learned policy still only sees the "
                    "FE latent, so it must reconstruct the board from memory.")
g.add_argument("--approximate-lambda", dest="approximate_lambda", action="store_true")
g.add_argument("--no-approximate-lambda", dest="approximate_lambda", action="store_false")
parser.set_defaults(approximate_lambda=False)
g.add_argument("--lambda1", type=float, default=0.05,
               help="short-horizon λ for the value λ-critic (reference uses 0.1). "
                    "Only active with --approximate-lambda.")
g.add_argument("--lambda2", type=float, default=0.75,
               help="long-horizon λ for the value λ-critic (reference uses 0.95). "
                    "Only active with --approximate-lambda.")
g.add_argument("--lambda-coef", dest="lambda_coef", type=float, default=1.0,
               help="weight on the value λ-discrepancy loss (Huber between the "
                    "two λ-critics), which shapes the shared FE. Only active "
                    "with --approximate-lambda.")
g.add_argument("--fake-onpolicy-loss", dest="fake_onpolicy_loss", action="store_true",
               help="clamp every V-trace importance ratio to 1.0, i.e. treat "
                    "replay data as on-policy (applies to the λ-critic and GVD "
                    "successor-feature losses alike)")
g.add_argument("--no-fake-onpolicy-loss", dest="fake_onpolicy_loss",
               action="store_false")
parser.set_defaults(fake_onpolicy_loss=True)
g.add_argument("--gvd", dest="gvd", action="store_true",
               help="enable the GVD successor-feature branches (reward-free "
                    "memory pressure; an agent trained with --gvd must be "
                    "reloaded with --gvd)")
g.add_argument("--no-gvd", dest="gvd", action="store_false")
parser.set_defaults(gvd=True)
g.add_argument("--gvd-coef", type=float, default=0.2, help="GVD discrepancy coefficient (default 1.0)")
g.add_argument("--gvd-features", type=int, default=16,
               help="random-projection width of the GVD feature map; total dim "
                    "is 1 (hit bit) + N (default 16)")
g.add_argument("--gvd-spatial", dest="gvd_spatial", action="store_true",
               help="use the action-localised SPATIAL hit cumulant "
                    "phi = [hit | (one_hot(a_prev)*hit) @ P] (P: N x gvd_features "
                    "fixed random projection) instead of the scalar hit bit. "
                    "Gives the memory a per-location target (restores the "
                    "pre-42a0a5f cumulant).")
parser.set_defaults(gvd_spatial=False)
g.add_argument("--gvd-raw", dest="gvd_raw", action="store_true",
               help="with --gvd-spatial, skip the random projection P and use "
                    "the RAW per-cell hit map phi = [hit | one_hot(a_prev)*hit] "
                    "(N+1 dims). The projection makes the spatial signal "
                    "near-zero-mean and board-independent in expectation "
                    "(washout); raw per-cell keeps a board-dependent target.")
parser.set_defaults(gvd_raw=False)
g.add_argument("--gvd-cumulant-diff", dest="gvd_cumulant_diff", action="store_true",
               help="build the SF cumulant as the temporal difference "
                    "f_t = phi(o_{t+1}) - phi(o_t) (Jaderberg-style SF-collapse "
                    "remedy) instead of raw phi(o_t); telescopes so the SF "
                    "predicts the remaining (board-dependent) ship cells.")
parser.set_defaults(gvd_cumulant_diff=False)
g.add_argument("--gvd-cumulant-scale", dest="gvd_cumulant_scale", type=float,
               default=1.0,
               help="scalar multiplier on the GVD SF cumulant. The raw hit "
                    "cumulant's SF grows to ~O(1/(1-gamma)) and its large TD "
                    "targets diverge the SF heads (sf1->negative). Set ~0.01 "
                    "(=1-gamma) to bound the SF to [0,1] (expected future "
                    "hit-rate) and stabilise; retune --gvd-coef (discrepancy "
                    "scales by scale^2). Default 1.0 = unchanged.")
g.add_argument("--gvd-lambda1", type=float, default=0.05)
g.add_argument("--gvd-lambda2", type=float, default=0.75)
g.add_argument("--gvd-sf-lr", type=float, default=1.8e-4)
g.add_argument("--gvd-stop-fe", dest="gvd_stop_fe", action="store_true",
               help="stop-gradient the FE latents feeding the GVD SF heads so "
                    "no GVD/SF gradient reaches the shared feature extractor "
                    "(stronger than --gvd-coef 0; SF heads still train)")
parser.set_defaults(gvd_stop_fe=False)
g.add_argument("--stop-actor-fe", dest="stop_actor_fe", action="store_true",
               help="stop-gradient the FE latents feeding the actor loss so the "
                    "shared FE is trained only by the value/critic (+GVD) losses "
                    "(SAC-AE recipe); removes actor-vs-critic gradient conflict "
                    "on the shared recurrent memory")
parser.set_defaults(stop_actor_fe=False)
g.add_argument("--stop-critic-fe", dest="stop_critic_fe", action="store_true",
               help="stop-gradient the FE latents feeding the critic losses "
                    "(main twin critic + lambda-critics + LD) so no value-head "
                    "gradient reaches the shared feature extractor; mirror of "
                    "--gvd-stop-fe, together they sever the whole value side "
                    "from the encoder (critic still trains)")
parser.set_defaults(stop_critic_fe=False)
g.add_argument("--vtrace-actor", dest="vtrace_actor", action="store_true",
               help="replace the SAC E_pi[Q] actor with an off-policy V-trace "
                    "(IMPALA-style) policy gradient: the actor maximises "
                    "rho*advantage*log pi(a|s) + alpha*H, with a V-trace value "
                    "baseline from the twin-Q critic. Removes the value-greedy "
                    "policy improvement (the deadly-triad divergence driver); "
                    "critic/GVD losses unchanged. Discrete only.")
parser.set_defaults(vtrace_actor=False)
g.add_argument("--vtrace-center-advantage", dest="vtrace_center_advantage",
               action="store_true",
               help="subtract the batch-mean advantage in the V-trace actor "
                    "(A2C/IMPALA variance reduction); removes the baseline-lag "
                    "offset so the policy gradient can sharpen the policy "
                    "instead of reinforcing sampled spread.")
parser.set_defaults(vtrace_center_advantage=False)
g.add_argument("--vtrace-normalize-advantage", dest="vtrace_normalize_advantage",
               action="store_true",
               help="full PPO-style advantage standardisation (A-mean)/std in "
                    "the V-trace actor (takes precedence over "
                    "--vtrace-center-advantage): centering alone leaves the "
                    "advantages at a tiny absolute scale under low-variance "
                    "returns, so the policy freezes at ~uniform.")
parser.set_defaults(vtrace_normalize_advantage=False)
g.add_argument("--ppo-clip-eps", dest="ppo_clip_eps", type=float, default=0.0,
               help="PPO clipped-surrogate epsilon for the V-trace actor "
                    "(0 = plain PG, no trust region). LOAD-BEARING: without it "
                    "the unclipped off-policy PG collapses the policy to a "
                    "deterministic board-blind firing order within a few steps "
                    "and the encoder's board memory dies with it.")
g.add_argument("--rho-bar", dest="rho_bar", type=float, default=1.05,
               help="V-trace rho truncation cap (default 1.05). rho=min(rho_bar, "
                    "pi/mu) is one-sided: on stale data the upside is clipped "
                    "but the downside is not, so E[rho]<1 attenuates the reward "
                    "correction. Raise it if replay data is heavily reused.")
g.add_argument("--c-bar", dest="c_bar", type=float, default=1.05,
               help="V-trace c truncation cap (default 1.05); see --rho-bar.")
g.add_argument("--mask-first-episode-only", dest="mask_first_episode_only",
               action="store_true",
               help="in the sequence losses, drop every step after the window's "
                    "first episode. With --episode-aligned-sampling a short "
                    "episode's window spills into the next episode's opening "
                    "steps, over-representing early/low-memory states and "
                    "biasing the FE gradient away from memory-rich late ones.")
parser.set_defaults(mask_first_episode_only=False)
g.add_argument("--dense-value-coef", dest="dense_value_coef", type=float, default=0.0,
               help="coefficient on the DENSE value loss: regress "
                    "V(s)=sum_a pi(a|s)*Q(s,a) against the same V-trace target "
                    "as the taken-action regression (expected-SARSA-style "
                    "projection). THE FIX for off-policy memory formation: a "
                    "per-action Q regressed only at a_t trains 1 of 100 output "
                    "columns per step, so the shared recurrent encoder gets a "
                    "~100x sparser gradient whose target column is chosen by the "
                    "behaviour policy; under a near-uniform actor that scatters "
                    "into noise. Applies to the lambda-critics AND the GVD SF "
                    "heads. 0 = off.")
g.add_argument("--no-sparse-value-loss", dest="sparse_value_loss",
               action="store_false",
               help="drop the taken-action regression and keep ONLY the dense "
                    "term (the form validated in the forward ladder). Safe with "
                    "--vtrace-actor (separate policy head); with SAC-style "
                    "extraction the per-action Q would lose its action "
                    "discrimination.")
parser.set_defaults(sparse_value_loss=True)
g.add_argument("--dense-discrepancy", dest="dense_discrepancy",
               action="store_true",
               help="compute the lambda-discrepancy (and the GVD discrepancy) "
                    "between the heads' EXPECTED values sum_a pi(a|s)*Q(s,a) "
                    "instead of at the executed action Q(s,a_t). Same sparsity "
                    "argument as --dense-value-coef, and it is what the "
                    "reference does: its discrepancy term is on the scalar V "
                    "heads, not per-action Q.")
parser.set_defaults(dense_discrepancy=False)
g.add_argument("--sac-critic-coef", dest="sac_critic_coef", type=float, default=1.0,
               help="weight on the main SAC critic's 1-STEP Bellman loss "
                    "(default 1.0; 0 drops it). A 1-step target V(s)=r+gV(s') is "
                    "satisfiable by a near-constant value with no history, so this "
                    "always-on head applies memoryless pressure to the shared "
                    "encoder every update; the reference has no equivalent (its "
                    "value regresses lam=0.95 near-Monte-Carlo returns).")
g.add_argument("--random-behaviour", dest="random_behaviour", action="store_true",
               help="collect with a uniform-random LEGAL behaviour policy "
                    "(order-invariant, board-revealing data, no learned-policy "
                    "structure). Pair with --actor-lr 0 --critic-lr 0 "
                    "--stop-actor-fe --stop-critic-fe to shape the FE by GVD "
                    "alone on random data (isolate reward-free memory-forcing).")
parser.set_defaults(random_behaviour=False)
g.add_argument("--alpha-anneal-final", dest="alpha_anneal_final", type=float,
               default=None, metavar="A_FINAL",
               help="linearly anneal the entropy temperature alpha from --alpha "
                    "(round 1) to this value (final round) across training; "
                    "None (default) = fixed alpha. Requires autotune off.")
g.add_argument("--batch-size", type=int, default=128)
g.add_argument("--sequence-length", type=int, default=80)
g.add_argument("--burn-in-length", type=int, default=5)
g.add_argument("--burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
               action="store_true",
               help="store the online carry per transition and initialise the "
                    "training burn-in from it instead of zeros (R2D2 "
                    "stored-state; enables shorter --burn-in-length; costs "
                    "carry_dim x buffer_size x 4 bytes extra)")
g.add_argument("--no-burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
               action="store_false")
parser.set_defaults(burn_in_from_stored_carry=True)
g.add_argument("--online-buffer-size", type=int, default=200_000)
g.add_argument("--episode-aligned-sampling", dest="episode_aligned_sampling",
               action="store_true",
               help="sample training windows that START at an episode start "
                    "(carry there is genuinely zero) instead of anywhere; roll "
                    "the FE from zero with no burn-in / no stored carry — "
                    "drift-free, no stored-carry staleness. Pair with "
                    "--no-burn-in-from-stored-carry --burn-in-length 0 and "
                    "--sequence-length >= the longest episode (rows*cols).")
parser.set_defaults(episode_aligned_sampling=False)
g.add_argument("--use-sac", dest="use_sac", action="store_true",
               help="whether or not to use a SAC entropy term for update"
                    "or just use entropy globally in loss")
g.add_argument("--no-use-sac", dest="use_sac", action="store_false")
parser.set_defaults(use_sac=False)

g = parser.add_argument_group("data collection")
g.add_argument("--collect-steps", type=int, default=100_000, help="total env steps to collect (default 100 000)")
g.add_argument("--collect-epsilon", type=float, default=1.0,
               help="epsilon-greedy rate for PROBE data collection: fraction of "
                    "shots taken uniformly at random over legal cells (perturbs "
                    "the fire order). Default 1.0 = fully random order = the "
                    "ORDER-INVARIANT probe (the honest board-memory metric; a "
                    "positional/temporal tape can't survive it). Set 0.0 to "
                    "probe under the agent's own order (tape-prone, inflated). "
                    "Only affects probe/eval data (collect_rollout), not agent "
                    "training or the critic/actor visualisations.")

g = parser.add_argument_group("probe training")
g.add_argument("--probe-steps", type=int, default=500_000, help="SGD steps (default 500 k)")
g.add_argument("--probe-lr", type=float, default=1e-4)
g.add_argument("--probe-hidden-dim", type=int, default=1024)
g.add_argument("--probe-batch-size", type=int, default=32)
g.add_argument("--probe-eval-interval", type=int, default=10,
               help="during agent training, train+score a lightweight probe every "
                    "N rounds and log it under probe_eval/ (0 disables; default 10)")
g.add_argument("--probe-eval-steps", type=int, default=100_000,
               help="SGD steps for each periodic probe eval (default 100 k)")
g.add_argument("--probe-eval-collect-steps", type=int, default=20_000,
               help="env steps collected per dataset for each periodic probe eval "
                    "(default 20 k; a train and a test set are collected)")
g.add_argument("--probe-eval-vis", dest="probe_eval_vis", action="store_true",
               help="also render episode/accuracy/retention images + an mp4 for each "
                    "periodic probe eval and log them to W&B (default on)")
g.add_argument("--no-probe-eval-vis", dest="probe_eval_vis", action="store_false")
parser.set_defaults(probe_eval_vis=True)

g = parser.add_argument_group("I/O & visualisation")
g.add_argument("--output-dir", default="./battleship_probe_output")
g.add_argument("--skip-train", action="store_true", help="load agent from output-dir")
g.add_argument("--skip-collect", action="store_true", help="load dataset from output-dir")
g.add_argument("--skip-probe", action="store_true", help="load probe from output-dir")
g.add_argument("--train-only", action="store_true",
               help="exit right after Phase-1 training (skip collect/probe/vis); "
                    "for stabilization sweeps that only watch training metrics")
g.add_argument("--save-checkpoint-at", type=int, default=0, metavar="ROUND",
               help="save a resumable training checkpoint (agent state + env "
                    "state + RNG key) after this round, then exit (0 = off)")
g.add_argument("--resume-from", default=None, metavar="PATH",
               help="resume Phase-1 training from a checkpoint written by "
                    "--save-checkpoint-at; continues with the CURRENT CLI "
                    "hyperparameters (so you can switch tau/lr/gvd_coef mid-run)")
g.add_argument("--vis-only", action="store_true",
               help="portable mode: load dataset.pkl + probe.pkl, render only "
                    "(no lambda-imitation / lambda-envs needed)")
g.add_argument("--mp4", action="store_true",
               help="save an mp4 (or gif fallback) of the longest collected episode")
g.add_argument("--vis-episodes", type=int, default=3, help="episodes to visualise")
g.add_argument("--vis-frames", type=int, default=8, help="frames per episode")

g = parser.add_argument_group("logging")
g.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
g.add_argument("--wandb-project", default="offline-lambda-battleship-probe", metavar="PROJECT",
               help="W&B project name (default: offline-lambda-battleship-probe)")
g.add_argument("--wandb-run-name", default=None, metavar="NAME", help="W&B run name (default: auto)")

# ── wandb sweep support ───────────────────────────────────────────────────────
#
# Under ``wandb agent`` the hyperparameters arrive via the sweep config, not the
# CLI: the agent passes underscore-style ``--name=value`` args that match neither
# the dashed flags nor the store_true/store_false pairs, so only there we
# tolerate unknown CLI args (they are validated via the sweep config below
# instead).  Outside a sweep, unknown arguments are a hard error.  Attach to
# the run the agent pre-created and override args from its config.  Mirrors
# battleship_sac_mc.py's sweep block.

if os.environ.get("WANDB_SWEEP_ID"):
    args, _ = parser.parse_known_args()
else:
    args = parser.parse_args()

_SWEEP_RUN = None
if os.environ.get("WANDB_SWEEP_ID"):
    import wandb as _wandb_mod

    _SWEEP_RUN = _wandb_mod.init()
    _SWEEP_ALIASES = {"use_gvd": "gvd"}
    _SWEEP_IGNORED = {"action_masked", "buffer_size"}  # not script knobs
    _ARG_TYPES = {a.dest: a.type for a in parser._actions if callable(a.type)}
    for _k, _v in dict(_SWEEP_RUN.config).items():
        _key = _SWEEP_ALIASES.get(_k, _k)
        if _key in _SWEEP_IGNORED:
            continue
        if not hasattr(args, _key):
            parser.error(f"sweep config: unknown parameter {_k!r}")
        if isinstance(_v, str) and _v.lower() in ("true", "false"):
            _v = _v.lower() == "true"
        elif _key in _ARG_TYPES:
            _v = _ARG_TYPES[_key](_v)
        setattr(args, _key, _v)
    # The agent owns exactly one run; log everything into it.
    args.wandb = True
    # Give every sweep run its own output dir (keyed by the unique W&B run id) so
    # several agents can share one machine without clobbering each other's
    # per-seed artifacts (agent_seed*.pkl, dataset_seed*.pkl, probe_seed*.pkl,
    # seed*/ image dirs).  Applied AFTER the config overrides so it also wins when
    # the sweep config pins ``output_dir``.
    args.output_dir = os.path.join(args.output_dir, f"run_{_SWEEP_RUN.id}")
    print(f"sweep run {_SWEEP_RUN.id}: output_dir → {args.output_dir}")

# ── always-needed imports ────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

# ── wandb (optional) ─────────────────────────────────────────────────────────
#
# Single-seed pipeline, so we log one run (cf. battleship_sac_mc.py's aggregated
# mode).  Agent training is keyed to ``env_interactions`` and probe training to
# ``probe_step`` via define_metric, so the two phases get separate x-axes and we
# never pass an explicit step= (which would clash across the two phases).  Vis-
# only mode is portable (no agent/env), so wandb stays off there.

_wandb = None
if args.wandb and not args.vis_only:
    try:
        import wandb as _wandb_mod

        _wandb = _wandb_mod
    except ImportError:
        sys.exit("wandb not installed.  Install with:  pip install wandb")

    _algo = ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else "")
    _wandb_cfg = {
            "env": "Battleship",
            "experiment": "board_probe",
            "algo": _algo,
            "rows": args.rows,
            "cols": args.cols,
            "ship_lengths": args.ship_lengths,
            "dense_reward": args.dense_reward,
            "terminal_bonus": args.terminal_bonus,
            "fe_lr": args.fe_lr,
            "grad_clip": args.grad_clip,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "alpha": args.alpha,
            "autotune_alpha": args.autotune_alpha,
            "target_entropy": args.target_entropy,
            "gamma": args.gamma,
            "tau": args.tau,
            "critic_layer_norm": args.critic_layer_norm,
            "seed": args.seed,
            "memory_type": args.memory_type,
            "memory_hidden_dim": args.memory_hidden_dim,
            "projection_dim": args.projection_dim,
            "paper_arch": args.paper_arch,
            "approximate_lambda": args.approximate_lambda,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "lambda_coef": args.lambda_coef,
            "use_gvd": args.gvd,
            "gvd_coef": args.gvd_coef,
            "gvd_features": args.gvd_features,
            "gvd_spatial": args.gvd_spatial,
            "gvd_raw": args.gvd_raw,
            "gvd_cumulant_diff": args.gvd_cumulant_diff,
            "gvd_cumulant_scale": args.gvd_cumulant_scale,
            "random_behaviour": args.random_behaviour,
            "gvd_lambda1": args.gvd_lambda1,
            "gvd_lambda2": args.gvd_lambda2,
            "gvd_sf_lr": args.gvd_sf_lr,
            "gvd_stop_fe": args.gvd_stop_fe,
            "stop_actor_fe": args.stop_actor_fe,
            "stop_critic_fe": args.stop_critic_fe,
            "vtrace_actor": args.vtrace_actor,
            "vtrace_center_advantage": args.vtrace_center_advantage,
            "vtrace_normalize_advantage": args.vtrace_normalize_advantage,
            "ppo_clip_eps": args.ppo_clip_eps,
            "rho_bar": args.rho_bar,
            "c_bar": args.c_bar,
            "mask_first_episode_only": args.mask_first_episode_only,
            "dense_value_coef": args.dense_value_coef,
            "sparse_value_loss": args.sparse_value_loss,
            "dense_discrepancy": args.dense_discrepancy,
            "sac_critic_coef": args.sac_critic_coef,
            "alpha_anneal_final": args.alpha_anneal_final,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "burn_in_length": args.burn_in_length,
            "burn_in_from_stored_carry": args.burn_in_from_stored_carry,
            "episode_aligned_sampling": args.episode_aligned_sampling,
            "online_buffer_size": args.online_buffer_size,
            "rounds": args.rounds,
            "train_steps": args.train_steps,
            "collect_steps": args.collect_steps,
            "collect_epsilon": args.collect_epsilon,
            "probe_steps": args.probe_steps,
            "probe_lr": args.probe_lr,
            "probe_hidden_dim": args.probe_hidden_dim,
            "probe_batch_size": args.probe_batch_size,
            "probe_eval_interval": args.probe_eval_interval,
            "probe_eval_steps": args.probe_eval_steps,
            "probe_eval_collect_steps": args.probe_eval_collect_steps,
            "use_sac": args.use_sac,
            "fake_onpolicy_loss": args.fake_onpolicy_loss,
            "num_seeds": args.num_seeds,
            "concurrent_seeds": args.concurrent_seeds,
            "final_return_window": args.final_return_window,
    }
    if _SWEEP_RUN is not None:
        # The sweep agent already created (and owns) the run — reuse it; just
        # merge our config so the dashboard shows the resolved hyperparameters.
        _SWEEP_RUN.config.update(_wandb_cfg, allow_val_change=True)
    else:
        _wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=_wandb_cfg,
        )
    _wandb.define_metric("env_interactions")
    _wandb.define_metric("agent/*", step_metric="env_interactions")
    _wandb.define_metric("probe_eval/*", step_metric="env_interactions")
    _wandb.define_metric("probe_step")
    _wandb.define_metric("probe/*", step_metric="probe_step")
    # Multi-seed: per-seed series under seed_i/, aggregates under agg/ and
    # probe_eval/agg/ and eval/agg/, all keyed to env_interactions.
    if args.num_seeds > 1:
        _wandb.define_metric("agg/*", step_metric="env_interactions")
        for _i in range(args.num_seeds):
            _wandb.define_metric(f"seed_{_i}/*", step_metric="env_interactions")

# ── probe MLP (pure JAX — no external deps) ──────────────────────────────────


def init_probe_params(key, carry_dim, n_out, hidden=1024):
    k1, k2, k3 = jax.random.split(key, 3)

    def layer(k, din, dout):
        return {
            "w": jax.random.normal(k, (din, dout)) * (2.0 / din) ** 0.5,
            "b": jnp.zeros(dout),
        }

    return {
        "l1": layer(k1, carry_dim, hidden),
        "l2": layer(k2, hidden, hidden),
        "l3": layer(k3, hidden, n_out),
    }


def probe_forward(params, x):
    x = jax.nn.relu(x @ params["l1"]["w"] + params["l1"]["b"])
    x = jax.nn.relu(x @ params["l2"]["w"] + params["l2"]["b"])
    return x @ params["l3"]["w"] + params["l3"]["b"]


def _auroc(scores, labels):
    """Threshold-free separability: P(score[ship] > score[water]).

    Rank-based (Mann-Whitney U); robust to the 14%/86% class imbalance that
    makes raw accuracy and recall@0.5 misleading.  0.5 = no signal, 1.0 =
    perfectly ranks ships above water.  NaN if a class is absent.
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ── shared rendering (module level so periodic probe-eval and the final ───────
#   visualisation phase build identical figures) ────────────────────────────
#
# These read the board dims (ROWS/COLS/N), CARRY_DIM and the colour constants
# as module globals — all defined before any figure is actually drawn.

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

WATER_COLOR = np.array([0.06, 0.12, 0.28])
SHIP_COLOR = np.array([0.15, 0.85, 0.35])
MISS_COLOR = np.array([0.55, 0.60, 0.70])
HIT_COLOR = np.array([0.95, 0.20, 0.20])
PRED_HIT_COLOR = np.array([1.0, 0.45, 0.10])   # orange: predicted fired + hit
PRED_MISS_COLOR = np.array([0.35, 0.65, 0.95])  # light blue: predicted fired + miss

# Colormaps for the colorbars: water→ship for P(ship), red→green for accuracy.
SHIP_CMAP = LinearSegmentedColormap.from_list("ship", [WATER_COLOR, SHIP_COLOR])
ACC_CMAP = LinearSegmentedColormap.from_list("acc", [(1.0, 0.0, 0.15), (0.0, 1.0, 0.15)])


def _board_legend(fig):
    """Shared legend: cell fill = ship belief, outline = the agent's shots.

    Shots are drawn as OUTLINES (not fills) so the probe's P(ship) underneath a
    fired cell stays visible — otherwise the ground-truth shot colour would hide
    whatever the probe actually predicts there.
    """
    handles = [
        mpatches.Patch(color=WATER_COLOR, label="fill: water (P≈0)"),
        mpatches.Patch(color=SHIP_COLOR, label="fill: ship (P≈1)"),
        mpatches.Patch(facecolor="none", edgecolor=MISS_COLOR, lw=1.6, label="truth outline: miss"),
        mpatches.Patch(facecolor="none", edgecolor=HIT_COLOR, lw=1.6, label="truth outline: hit"),
        mpatches.Patch(facecolor="none", edgecolor=PRED_HIT_COLOR, lw=1.6, ls="--",
                       label="probe outline (dashed): pred. fired, hit"),
        mpatches.Patch(facecolor="none", edgecolor=PRED_MISS_COLOR, lw=1.6, ls="--",
                       label="probe outline (dashed): pred. fired, miss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))


def to_grid(values):
    """Reshape a length-(ROWS*COLS) vector to the board grid (row-major)."""
    return np.asarray(values).reshape(ROWS, COLS)


def render(ax, ship_grid, fired_grid, title="", pred_fired=False):
    """Fill = ship belief (water→green); fired cells drawn as outlines on top.

    The outline does NOT overwrite the cell colour, so on the probe panel you can
    see whether the probe's belief at a fired cell matches the shot result
    (e.g. green fill under a 'fired' outline = correctly recalled hit).

    ``pred_fired=False`` (truth panel): ``fired_grid`` holds the true hit/miss
    code (1=miss, 2=hit) and outlines are coloured grey/red accordingly (solid).
    ``pred_fired=True`` (probe panel): ``fired_grid`` holds P(fired); any cell
    with P>0.5 gets a DASHED outline — the probe's reconstruction of *which*
    cells were fired at — coloured by the predicted shot result.  A fired cell
    is a hit iff a ship occupies it, so the probe's own P(ship) at that cell IS
    its hit/miss prediction: orange = predicted hit (P(ship)>0.5), light blue =
    predicted miss.
    """
    img = np.empty((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            v = float(np.nan_to_num(ship_grid[r, c]))
            img[r, c] = WATER_COLOR * (1 - v) + SHIP_COLOR * v
    ax.imshow(img, interpolation="nearest", aspect="equal")
    for r in range(ROWS):
        for c in range(COLS):
            fv = fired_grid[r, c]
            ls = "solid"
            if pred_fired:
                if fv > 0.5:
                    hit_pred = float(np.nan_to_num(ship_grid[r, c])) > 0.5
                    edge = PRED_HIT_COLOR if hit_pred else PRED_MISS_COLOR
                    ls = (0, (3, 1.5))  # dashed: prediction, not ground truth
                else:
                    continue
            else:
                if fv == 1 or fv == 2:
                    edge = HIT_COLOR if fv == 2 else MISS_COLOR
                else:
                    continue
            # inset slightly so adjacent fired cells don't double their borders
            ax.add_patch(mpatches.Rectangle(
                (c - 0.38, r - 0.38), 0.76, 0.76, fill=False, edgecolor=edge,
                lw=1.3, linestyle=ls,
            ))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_acc(ax, grid, title):
    img = np.empty((ROWS, COLS, 3))
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            img[r, c] = [0.5, 0.5, 0.5] if np.isnan(v) else [1.0 - v, v, 0.15]
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _fig_episode(probe_params, ds_c, ds_b, ds_hm, ds_eb, ve_idx, out_path, tag_str,
                 vis_frames, probs_all=None):
    """Render one stored episode (truth vs probe) as a multi-frame PNG → out_path.

    ``probs_all`` (host sigmoid probabilities for the *full* ``ds_c``, shape
    ``(len(ds_c), 2N)``) lets the caller skip the in-helper GPU forward pass —
    used by the multi-seed async-overlap path so rendering is pure CPU.
    """
    start, end = int(ds_eb[ve_idx]), int(ds_eb[ve_idx + 1])
    ep_b, ep_hm = ds_b[start:end], ds_hm[start:end]
    ep_len = end - start
    if probs_all is not None:
        preds = np.asarray(probs_all)[start:end]
    else:
        preds = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ds_c[start:end]))))
    ship_preds, fired_preds = preds[:, :N], preds[:, N:]
    nf = min(vis_frames, ep_len)
    idxs = np.linspace(0, ep_len - 1, nf, dtype=int)
    fig, axes = plt.subplots(2, nf, figsize=(2.4 * nf, 6))
    if nf == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"Board Probe — {tag_str}  episode {ve_idx + 1}  (len={ep_len})", fontsize=10)
    for j, idx in enumerate(idxs):
        render(axes[0, j], to_grid(ep_b[idx]), to_grid(ep_hm[idx]), f"True  t={idx}")
        render(axes[1, j], to_grid(ship_preds[idx]), to_grid(fired_preds[idx]),
               f"Pred  t={idx}", pred_fired=True)
    axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
    axes[1, 0].set_ylabel("Probe", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 0.94, 1])
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SHIP_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("P(ship)  —  truth is 0/1, probe is continuous", fontsize=8)
    _board_legend(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fig_accuracy(probe_params, t_carries, t_boards, t_hm, out_path, tag_str, probs_all=None):
    """Per-cell accuracy (unfired=inference, fired=retention) → out_path."""
    if probs_all is not None:
        probs = np.asarray(probs_all)[:, :N]
    else:
        probs = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(t_carries))))[:, :N]
    preds = (probs > 0.5).astype(np.float32)
    targets = np.asarray(t_boards).astype(np.float32)
    correct = (preds == targets).astype(np.float32)
    fired = np.asarray(t_hm) != 0
    unfired = ~fired

    def _pcacc(mask):
        m = mask.astype(np.float32)
        den = m.sum(0)
        g = np.where(den > 0, (correct * m).sum(0) / np.maximum(den, 1.0), np.nan)
        return to_grid(g)

    def _a(mask):
        m = mask.astype(np.float32)
        return float((correct * m).sum() / max(m.sum(), 1.0))

    fig, (ax_u, ax_f) = plt.subplots(1, 2, figsize=(9, 5.0))
    _draw_acc(ax_u, _pcacc(unfired), f"Unfired — inference\n{_a(unfired):.1%}")
    _draw_acc(ax_f, _pcacc(fired), f"Fired — retention\n{_a(fired):.1%}")
    fig.suptitle(f"Per-Cell Probe Accuracy (test) — {tag_str}   overall {float(correct.mean()):.1%}",
                 fontsize=11)
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=ACC_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=(ax_u, ax_f), fraction=0.046, pad=0.04)
    cbar.set_label("per-cell accuracy (grey = never in this category)", fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fig_retention(probe_params, t_carries, t_boards, t_hm, t_eb, out_path, tag_str, probs_all=None):
    """Fired-cell mean P(ship) vs steps-since-fired → out_path (None if no data)."""
    if probs_all is not None:
        probs = np.asarray(probs_all)[:, :N]
    else:
        probs = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(t_carries))))[:, :N]
    targets = np.asarray(t_boards).astype(np.float32)
    hm = np.asarray(t_hm)
    ship_frac = float((targets == 1).mean())
    fired = hm != 0
    fired_auroc = _auroc(probs[fired], targets[fired])

    rec_list, prob_list, tgt_list = [], [], []
    for i in range(len(t_eb) - 1):
        s, e = int(t_eb[i]), int(t_eb[i + 1])
        if e - s <= 0:
            continue
        hme, prob, tgt = hm[s:e], probs[s:e], targets[s:e]
        fired_ep = hme != 0
        L = e - s
        any_fired = fired_ep.any(0)
        fire_step = np.where(any_fired, fired_ep.argmax(0), 0)
        rec = np.arange(L)[:, None] - fire_step[None, :]
        valid = fired_ep & (rec >= 0)
        rec_list.append(rec[valid])
        prob_list.append(prob[valid])
        tgt_list.append(tgt[valid])

    if not (rec_list and sum(len(r) for r in rec_list)):
        return None
    rec_all = np.concatenate(rec_list)
    prob_all = np.concatenate(prob_list)
    tgt_all = np.concatenate(tgt_list)
    maxr = int(rec_all.max())

    def _mean_by_rec(sel):
        r, p = rec_all[sel], prob_all[sel]
        s_ = np.bincount(r, weights=p, minlength=maxr + 1)
        n_ = np.bincount(r, minlength=maxr + 1)
        return np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan), n_

    hit_prob, n_hit = _mean_by_rec(tgt_all == 1)
    miss_prob, n_miss = _mean_by_rec(tgt_all == 0)
    x = np.arange(maxr + 1)

    def _xy(y, n, thr=20):
        m = n >= thr
        return x[m], y[m]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(*_xy(hit_prob, n_hit), "-o", ms=3, color=HIT_COLOR,
            label="hit cells (true ship): mean P(ship)")
    ax.plot(*_xy(miss_prob, n_miss), "-o", ms=3, color="#4c72b0",
            label="miss cells (true water): mean P(ship)")
    ax.axhline(ship_frac, ls=":", lw=1, color="grey", label=f"prior P(ship)={ship_frac:.2f}")
    ax.set_xlabel("steps since the cell was fired")
    ax.set_ylabel("mean predicted P(ship)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Retention horizon — {tag_str}  (memory={CARRY_DIM}-dim, AUROC={fired_auroc:.2f})\n"
        f"threshold-free: gap between hit/miss curves = decodable memory of the cell",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _fig_movie(probe_params, ep_c, ep_b, ep_hm, out_path_base, tag_str, movie_tag, fps=10,
               probs_all=None):
    """Animate one episode (truth vs probe) → mp4 (or gif fallback); returns path.

    ``probs_all`` (host sigmoid probabilities for ``ep_c``) skips the GPU forward
    pass for the async-overlap path.
    """
    from matplotlib.animation import FuncAnimation

    ep_len = len(ep_b) if probs_all is not None else len(ep_c)
    if ep_len == 0:
        return None
    if probs_all is not None:
        ep_preds = np.asarray(probs_all)
    else:
        ep_preds = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(ep_c))))
    ep_ship, ep_fired = ep_preds[:, :N], ep_preds[:, N:]

    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(10, 6))
    _sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=SHIP_CMAP)
    _sm.set_array([])
    fig.colorbar(_sm, ax=(ax_t, ax_p), fraction=0.025, pad=0.02).set_label("P(ship)", fontsize=8)
    _board_legend(fig)

    def _update(frame):
        ax_t.clear()
        ax_p.clear()
        render(ax_t, to_grid(ep_b[frame]), to_grid(ep_hm[frame]), "Ground Truth")
        render(ax_p, to_grid(ep_ship[frame]), to_grid(ep_fired[frame]),
               "Probe Prediction", pred_fired=True)
        fig.suptitle(f"Board Probe — {tag_str}  ({movie_tag})   t = {frame}/{ep_len}", fontsize=11)

    anim = FuncAnimation(fig, _update, frames=ep_len, interval=100)
    out = out_path_base + ".mp4"
    try:
        anim.save(out, writer="ffmpeg", fps=fps)
    except Exception:
        out = out_path_base + ".gif"
        anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    return out


# ── critic-Q / actor-π heatmaps (agent introspection, not the probe) ──────────
#
# These visualise the *agent's own* value & policy over the board, read straight
# from the critic / actor (via debug_fns.predict_qpi) along a deterministic
# actor-greedy rollout — distinct from the probe panels (which decode the board
# from memory).  Image: rows = [ground truth, critic Q, actor π] × frame cols.
# Video: ground truth | critic Q | actor π, left→right, animated over the episode.

CRITIC_CMAP = "viridis"   # Q-values (arbitrary scale → per-episode normalised)
ACTOR_CMAP = "magma"      # π(a) ∈ [0, 1]


FIRED_GREY = np.array([0.30, 0.30, 0.34])   # greyed-out fill for already-fired cells


def _heatmap(ax, grid, title, cmap, vmin, vmax, fired_grid=None):
    """imshow a board-shaped value grid.

    Already-fired cells are GREYED OUT (their Q/π is unconstrained by any loss —
    masked out of V/policy/CQL — so its value is meaningless); the shot result is
    still shown as a coloured outline (red=hit, grey=miss)."""
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(FIRED_GREY)                  # NaN cells render as grey
    grid = np.asarray(grid, dtype=float)
    if fired_grid is not None:
        grid = np.where(np.asarray(fired_grid) != 0, np.nan, grid)  # blank fired cells
    im = ax.imshow(grid, cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    if fired_grid is not None:
        for r in range(ROWS):
            for c in range(COLS):
                fv = fired_grid[r, c]
                if fv == 1 or fv == 2:
                    ax.add_patch(mpatches.Rectangle(
                        (c - 0.42, r - 0.42), 0.84, 0.84, fill=False,
                        edgecolor=(HIT_COLOR if fv == 2 else MISS_COLOR), lw=1.1))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _qpi_norm(ep_q, ep_probs, ep_hm=None):
    """Per-episode colour ranges for the Q and π heatmaps, computed over the
    LEGAL (unfired) cells only — fired cells are greyed out and their
    unconstrained Q would otherwise wash out the meaningful range."""
    q = np.asarray(ep_q, dtype=float)
    p = np.asarray(ep_probs, dtype=float)
    if ep_hm is not None:
        legal = np.asarray(ep_hm) == 0
        q = np.where(legal, q, np.nan)
        p = np.where(legal, p, np.nan)
    qmin, qmax = float(np.nanmin(q)), float(np.nanmax(q))
    if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax - qmin < 1e-6:
        qmin, qmax = (0.0, 1e-6) if not np.isfinite(qmin) else (qmin, qmin + 1e-6)
    pmax = float(np.nanmax(p)) if np.isfinite(np.nanmax(p)) else 1e-6
    pmax = max(pmax, 1e-6)
    return qmin, qmax, pmax


def _fig_qpi_episode(ep_b, ep_hm, ep_q, ep_probs, out_path, tag_str, vis_frames):
    """Multi-frame PNG: rows = [GT, Critic Q, Actor π], cols = sampled frames."""
    ep_len = len(ep_b)
    if ep_len == 0:
        return None
    idxs = np.unique(np.linspace(0, ep_len - 1, min(vis_frames, ep_len)).astype(int))
    F = len(idxs)
    qmin, qmax, pmax = _qpi_norm(ep_q, ep_probs, ep_hm)
    fig, axes = plt.subplots(3, F, figsize=(2.0 * F + 1.4, 6.6), squeeze=False)
    im_c = im_a = None
    for j, t in enumerate(idxs):
        render(axes[0, j], to_grid(ep_b[t]), to_grid(ep_hm[t]), f"t={t}")
        im_c = _heatmap(axes[1, j], to_grid(ep_q[t]), f"t={t}",
                        CRITIC_CMAP, qmin, qmax, fired_grid=to_grid(ep_hm[t]))
        im_a = _heatmap(axes[2, j], to_grid(ep_probs[t]), f"t={t}",
                        ACTOR_CMAP, 0.0, pmax, fired_grid=to_grid(ep_hm[t]))
    for row, label in enumerate(("Ground truth", "Critic Q", "Actor π")):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.colorbar(im_c, ax=axes[1, :].tolist(), fraction=0.012, pad=0.01).set_label("Q", fontsize=7)
    fig.colorbar(im_a, ax=axes[2, :].tolist(), fraction=0.012, pad=0.01).set_label("π(a)", fontsize=7)
    fig.suptitle(f"Critic value & actor policy — {tag_str}", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _fig_qpi_movie(ep_b, ep_hm, ep_q, ep_probs, out_path_base, tag_str, fps=10):
    """Animate GT | Critic Q | Actor π (left→right) over one episode → mp4/gif."""
    from matplotlib.animation import FuncAnimation

    ep_len = len(ep_b)
    if ep_len == 0:
        return None
    qmin, qmax, pmax = _qpi_norm(ep_q, ep_probs, ep_hm)
    fig, (ax_g, ax_c, ax_a) = plt.subplots(1, 3, figsize=(13, 5))
    sm_c = ScalarMappable(norm=Normalize(qmin, qmax), cmap=CRITIC_CMAP); sm_c.set_array([])
    sm_a = ScalarMappable(norm=Normalize(0.0, pmax), cmap=ACTOR_CMAP); sm_a.set_array([])
    fig.colorbar(sm_c, ax=ax_c, fraction=0.046, pad=0.02).set_label("Q", fontsize=8)
    fig.colorbar(sm_a, ax=ax_a, fraction=0.046, pad=0.02).set_label("π(a)", fontsize=8)

    def _update(frame):
        ax_g.clear(); ax_c.clear(); ax_a.clear()
        render(ax_g, to_grid(ep_b[frame]), to_grid(ep_hm[frame]), "Ground Truth")
        _heatmap(ax_c, to_grid(ep_q[frame]), "Critic Q", CRITIC_CMAP, qmin, qmax,
                 fired_grid=to_grid(ep_hm[frame]))
        _heatmap(ax_a, to_grid(ep_probs[frame]), "Actor π", ACTOR_CMAP, 0.0, pmax,
                 fired_grid=to_grid(ep_hm[frame]))
        fig.suptitle(f"Critic Q & Actor π — {tag_str}   t = {frame}/{ep_len}", fontsize=12)

    anim = FuncAnimation(fig, _update, frames=ep_len, interval=100)
    out = out_path_base + ".mp4"
    try:
        anim.save(out, writer="ffmpeg", fps=fps)
    except Exception:
        out = out_path_base + ".gif"
        anim.save(out, writer="pillow", fps=fps)
    plt.close(fig)
    return out


# ── output paths ─────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)
agent_path = os.path.join(args.output_dir, "agent.pkl")
dataset_path = os.path.join(args.output_dir, "dataset.pkl")
test_dataset_path = os.path.join(args.output_dir, "test_dataset.pkl")
probe_path = os.path.join(args.output_dir, "probe.pkl")


# ════════════════════════════════════════════════════════════════════════════
#  Full pipeline (phases 1-3) — skipped entirely in --vis-only
# ════════════════════════════════════════════════════════════════════════════

if not args.vis_only:
    import optax
    from tqdm.rich import tqdm

    try:
        from lambda_envs.envs.battleship import Battleship
    except ImportError:
        sys.exit("lambda-envs required.  pip install lambda-envs")

    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import (
        battleship_projection,
        create_iqlearn_from_env,
        env_spec_from_gymnax,
    )

    # ── env setup ──────────────────────────────────────────────────────────────
    #
    # The previous action reaches the feature extractor via
    # ``use_prev_action=True`` (fed as an explicit input to the projection and
    # threaded next to the carry; see RecurrentFeatureExtractor), so the env is
    # used unwrapped.  The memory
    # then sees *which* cell was fired and *whether* it hit, which is exactly
    # what it needs to reconstruct the hidden board.

    ship_lengths = tuple(int(x) for x in args.ship_lengths.split(",") if x.strip())
    if any(l > max(args.rows, args.cols) for l in ship_lengths):
        sys.exit(f"ship length {max(ship_lengths)} does not fit on a "
                 f"{args.rows}x{args.cols} board; pass shorter --ship-lengths.")
    N = args.rows * args.cols  # number of actions / board cells

    if args.full_obs:
        # DIAGNOSTIC env: obs = [last_hit_miss(1) | legal_mask(N) | hits_misses(N)].
        # The trailing per-cell hits_misses (0/1/2 → scaled) makes the obs Markov
        # so the policy needs no memory to play hunt/target.
        import gymnax.environments.spaces as _gspaces

        class _BattleshipFullObs(Battleship):
            def observation_space(self, params):
                return _gspaces.Box(0, 1, (1 + 2 * N,))

            def get_obs(self, state, params=None):
                base = super().get_obs(state, params)            # [hit(1)|mask(N)]
                hm = state.hits_misses.reshape(-1).astype(float) / 2.0  # {0,.5,1}
                return jnp.concatenate([base, hm], axis=-1)

        env = _BattleshipFullObs(rows=args.rows, cols=args.cols,
                                 dense_reward=args.dense_reward,
                                 terminal_bonus=args.terminal_bonus,
                                 ship_lengths=ship_lengths)
    else:
        env = Battleship(rows=args.rows, cols=args.cols,
                         dense_reward=args.dense_reward,
                         terminal_bonus=args.terminal_bonus,
                         ship_lengths=ship_lengths)
    env_params = env.default_params
    spec = env_spec_from_gymnax(env, env_params)
    ROWS, COLS = args.rows, args.cols  # board dims (used by the shared renderers)
    if args.full_obs:
        # FE sees [hit_bit | hits_misses grid]; mask from the legal-mask slice;
        # gvd_feature_fn still reads o[...,:1] (the last-hit bit) below.
        obs_fn = lambda o: jnp.concatenate([o[..., 0:1], o[..., 1 + N:1 + 2 * N]], axis=-1)
        mask_fn = lambda o: o[..., 1:1 + N]
    else:
        # obs layout: [last_hit_miss(1) | mask(N)]
        obs_fn = lambda o: o[..., 0:1]
        mask_fn = lambda o: o[..., 1:1 + N]

    expert_data = {
        "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((1, 1), dtype=jnp.float32),
    }

    # GVD feature map φ(o_t, a_{t-1}) = [hit bit | (one_hot(a_{t-1})·hit)·P] —
    # an action-localised *spatial* hit cumulant (see battleship_sac_mc.py for
    # the rationale).  Identical construction (fixed jax.random.key(0)
    # projection, independent of --seed) so agents trained there probe the
    # same way here.
    if args.gvd:
        _GVD_P = jax.random.normal(
            jax.random.key(0), (N, args.gvd_features)
        ) / jnp.sqrt(args.gvd_features)

        def gvd_feature_fn(o, a_prev):
            hit = o[..., :1]  # result of a_prev (0 at episode starts)
            if args.gvd_spatial:
                loc = jax.nn.one_hot(a_prev[..., 0].astype(jnp.int32), N)
                if args.gvd_raw:
                    # raw per-cell hit map (N+1 dims), no projection washout
                    return jnp.concatenate([hit, loc * hit], axis=-1)
                # action-localised spatial hit cumulant, projected to
                # gvd_features dims: [hit | (one_hot(a_prev)*hit) @ P].
                return jnp.concatenate([hit, (loc * hit) @ _GVD_P], axis=-1)
            return hit
    else:
        gvd_feature_fn = None

    # ── carry helper ─────────────────────────────────────────────────────────
    #
    # CARRY_DIM is the memory part (the probe input).  The prev-action one-hot
    # (use_prev_action=True) is threaded separately, not packed into the carry,
    # so the agent carry is exactly CARRY_DIM wide.

    if args.memory_type == "identity":
        CARRY_DIM = 0
    elif args.memory_type == "lstm":
        CARRY_DIM = 2 * args.memory_hidden_dim
    else:
        CARRY_DIM = args.memory_hidden_dim

    _memoryless = (CARRY_DIM == 0)
    if _memoryless and not args.full_obs:
        sys.exit("Probe needs recurrent memory (--memory-type rnn/gru/lstm), not identity.")
    # Cadence for the critic-Q / actor-π heatmaps.  These only need predict_qpi
    # (NOT the board probe / a recurrent carry), so they render even in the
    # memoryless diagnostic — captured here before the probe is disabled below.
    _qpi_interval = args.probe_eval_interval if args.probe_eval_interval > 0 else 10
    if _memoryless:
        # Feedforward-value diagnostic (full-obs, no memory needed): the board
        # probe needs a carry to read, so it's skipped — but the critic/actor
        # heatmaps + critic-greedy metrics still run.
        print("[memoryless diagnostic] identity memory + full-obs: board-probe "
              "phases disabled; training + eval + critic/actor heatmaps only.")
        args.probe_eval_interval = 0

    # The carry is the memory state only; the prev-action one-hot is threaded
    # separately (not packed into the carry).
    AGENT_CARRY_DIM = CARRY_DIM

    # Network architecture.  --paper-arch swaps the projection + head sizes
    # for the original lambda-discrepancy BattleShipActorCriticRNN: a custom
    # hit-skip-connection embedding (width H = --memory-hidden-dim) feeding the
    # GRU, with single-hidden-layer actor/critic heads of width H.  Otherwise
    # the default LinearProjection(--projection-dim) + (256, 256) heads.
    if args.paper_arch:
        H = args.memory_hidden_dim
        # With no recurrent cell (identity), add the third Dense(H)->relu so the
        # embedding matches the paper's memoryless BattleShipActorCritic; with a
        # GRU the cell provides that depth (BattleShipActorCriticRNN).
        projection_arg = battleship_projection(H, extra_layer=_memoryless)
        actor_dims = (H,)
        critic_dims = (H,)
    else:
        projection_arg = args.projection_dim if args.projection_dim > 0 else None
        actor_dims = (256, 256)
        critic_dims = (256, 256)

    def zero_carry():
        return jnp.zeros((AGENT_CARRY_DIM,), dtype=jnp.float32)

    def zero_prev_action():
        return jnp.zeros((N,), dtype=jnp.float32)  # use_prev_action=True here

    # ── hyperparameters ──────────────────────────────────────────────────────

    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=args.online_buffer_size,
        target_entropy=args.target_entropy,
        grad_clip=args.grad_clip,
        fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        lambda_critic_lr=1e-4, alpha_lr=1e-4,
        alpha=args.alpha, autotune_alpha=args.autotune_alpha,
        batch_size=args.batch_size, gamma=args.gamma, tau=args.tau,
        lambda1=args.lambda1, lambda2=args.lambda2,
        c_bar=args.c_bar, rho_bar=args.rho_bar, lambda_truncation=20,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        lambda_coef=args.lambda_coef, fake_onpolicy_loss=args.fake_onpolicy_loss,
        gvd_coef=args.gvd_coef,
        gvd_lambda1=args.gvd_lambda1,
        gvd_lambda2=args.gvd_lambda2,
        gvd_sf_lr=args.gvd_sf_lr,
        gvd_stop_fe=args.gvd_stop_fe,
        stop_actor_fe=args.stop_actor_fe,
        stop_critic_fe=args.stop_critic_fe,
        vtrace_actor=args.vtrace_actor,
        vtrace_center_advantage=args.vtrace_center_advantage,
        vtrace_normalize_advantage=args.vtrace_normalize_advantage,
        ppo_clip_eps=args.ppo_clip_eps,
        mask_first_episode_only=args.mask_first_episode_only,
        dense_value_coef=args.dense_value_coef,
        sparse_value_loss=args.sparse_value_loss,
        dense_discrepancy=args.dense_discrepancy,
        sac_critic_coef=args.sac_critic_coef,
        gvd_cumulant_diff=args.gvd_cumulant_diff,
        gvd_cumulant_scale=args.gvd_cumulant_scale,
        random_behaviour=args.random_behaviour,
        episode_aligned_sampling=args.episode_aligned_sampling,
    )

    # Battleship episodes end after at most rows*cols shots (legal-action
    # mask exhausts the board) — scan only that far, not the env's 1000.
    _MAX_STEPS = min(
        int(env_params.max_steps_in_episode), args.rows * args.cols + 1
    )

    # ── multi-seed helpers ─────────────────────────────────────────────────────
    #
    # Concurrent seeds are trained in one vmapped+jitted kernel: per-seed agent
    # states are stacked along a leading axis, vmapped over, then split back out.
    # Mirrors battleship_sac_mc.py:754-762.

    def _stack_states(states):
        """Stack a list of per-seed pytrees along a new leading axis."""
        return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

    def _unstack_state(batched, j):
        """Pull seed ``j``'s pytree out of a leading-axis-batched state."""
        return jax.tree.map(lambda x: x[j], batched)

    def _split_each(keys):
        """Split a batch of PRNG keys, returning two batches (carry, fresh)."""
        out = jax.vmap(lambda k: jax.random.split(k))(keys)
        return out[:, 0], out[:, 1]

    def _agg(values, prefix):
        """mean / std / sterr (+ band edges) of per-seed scalars (NaN-safe).

        std is the sample std (ddof=1; →0 when n≤1) and sterr = std/√n, so the
        aggregates are the natural across-seed uncertainty bars.  NaN entries
        (e.g. AUROC when a class is absent for a seed) are dropped.

        Also logs precomputed band edges so a plain W&B line panel can show the
        error bands directly (no Vega calculate transform needed):
        ``lo_sd``/``hi_sd`` = mean ∓ std, ``lo_se``/``hi_se`` = mean ∓ sterr.
        """
        a = np.asarray(values, dtype=np.float64).ravel()
        a = a[~np.isnan(a)]
        n = a.size
        mean = float(a.mean()) if n else float("nan")
        std = float(a.std(ddof=1)) if n > 1 else 0.0
        sterr = std / np.sqrt(n) if n > 1 else 0.0
        return {
            f"{prefix}/mean": mean,
            f"{prefix}/std": std,
            f"{prefix}/sterr": sterr,
            f"{prefix}/lo_sd": mean - std,
            f"{prefix}/hi_sd": mean + std,
            f"{prefix}/lo_se": mean - sterr,
            f"{prefix}/hi_se": mean + sterr,
        }

    def _build_agent(seed_val):
        return create_iqlearn_from_env(
            spec, expert_data, buffer_size=1, hp=hp,
            projection=projection_arg,
            memory_type=args.memory_type,
            memory_hidden_dim=args.memory_hidden_dim,
            actor_dims=actor_dims,
            critic_dims=critic_dims,
            lambda1_critic_dims=(64,64,64),
            lambda2_critic_dims=(64,64,64),
            train_steps=args.train_steps,
            approximate_lambda=args.approximate_lambda,
            use_prev_action=True,
            critic_layer_norm=args.critic_layer_norm,
            obs_fn=obs_fn, mask_fn=mask_fn,
            burn_in_from_stored_carry=args.burn_in_from_stored_carry,
            use_gvd=args.gvd, gvd_feature_fn=gvd_feature_fn,
            gvd_sf_dims=(256, 256),
            debug=True, seed=seed_val,
            use_sac=args.use_sac
        )

    # ── evaluation helper ────────────────────────────────────────────────────

    def _make_evaluate(fns):
        @partial(jax.jit, static_argnames=["n_episodes"])
        def _evaluate(agent_state, rng_key, n_episodes=10):
            def run_episode(key):
                key, rk = jax.random.split(key)
                obs, env_st = env.reset(rk, env_params)
                carry = zero_carry()
                prev_action = zero_prev_action()
                def step_fn(s, _):
                    obs, env_st, carry, prev_action, key, ret, done, steps = s
                    key, sk, ek = jax.random.split(key, 3)
                    raw, nc = fns.predict(
                        agent_state, obs, carry, sk, deterministic=True,
                        prev_action=prev_action,
                    )
                    action = jnp.round(raw).astype(jnp.int32)
                    nobs, nst, rew, d, _ = env.step(ek, env_st, action, env_params)
                    npa = fns.encode_action(jnp.atleast_1d(raw))
                    ret = ret + rew * (1.0 - done)
                    steps = steps + (1.0 - done)   # steps taken until first clear
                    done = jnp.maximum(done, d.astype(jnp.float32))
                    npa = jnp.where(done > 0, jnp.zeros_like(npa), npa)
                    return (nobs, nst, nc, npa, key, ret, done, steps), None
                init = (obs, env_st, carry, prev_action, key,
                        jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
                (_, _, _, _, _, ep_ret, ep_done, ep_steps), _ = jax.lax.scan(
                    step_fn, init, length=_MAX_STEPS)
                return ep_ret, ep_steps, ep_done   # done=1 iff cleared within horizon
            keys = jax.random.split(rng_key, n_episodes)
            rets, steps, dones = jax.vmap(run_episode)(keys)
            # steps-to-clear (the policy-quality metric; dense return saturates).
            return jnp.mean(rets), jnp.mean(steps), jnp.mean(dones)
        return _evaluate

    def _make_evaluate_critic(debug_fns):
        """Critic-greedy eval: act by argmax_a Q(s,a) over LEGAL cells (via
        debug_fns.predict_qpi), same scan/metrics as the actor-greedy eval.
        Comparing the two localises Problem #2: if critic-greedy plays well but
        actor-greedy doesn't, it's an actor-extraction gap; if both are random,
        the critic's per-action ranking itself is wrong."""
        @partial(jax.jit, static_argnames=["n_episodes"])
        def _evaluate(agent_state, rng_key, n_episodes=10):
            def run_episode(key):
                key, rk = jax.random.split(key)
                obs, env_st = env.reset(rk, env_params)
                carry = zero_carry()
                prev_action = zero_prev_action()
                def step_fn(s, _):
                    obs, env_st, carry, prev_action, key, ret, done, steps, spread, rng = s
                    key, ek = jax.random.split(key)
                    q, _probs, nc = debug_fns.predict_qpi(
                        agent_state, obs, carry, prev_action=prev_action)
                    legalb = mask_fn(obs) > 0
                    nl = jnp.maximum(jnp.sum(legalb), 1.0)
                    # std of Q across LEGAL actions at this state (flatness of the
                    # per-action ranking; ≈0 ⇒ Q is a per-step constant).
                    qm = jnp.sum(jnp.where(legalb, q, 0.0)) / nl
                    qstd = jnp.sqrt(jnp.sum(jnp.where(legalb, (q - qm) ** 2, 0.0)) / nl)
                    # range = best-legal − worst-legal Q (the actor's max advantage)
                    qrng = (jnp.max(jnp.where(legalb, q, -jnp.inf))
                            - jnp.min(jnp.where(legalb, q, jnp.inf)))
                    action = jnp.argmax(jnp.where(legalb, q, -jnp.inf)).astype(jnp.int32)
                    npa = jax.nn.one_hot(action, N)   # executed action → next prev-action input
                    nobs, nst, rew, d, _ = env.step(ek, env_st, action, env_params)
                    ret = ret + rew * (1.0 - done)
                    steps = steps + (1.0 - done)
                    spread = spread + jnp.stack([qstd, qrng]) * (1.0 - done)
                    done = jnp.maximum(done, d.astype(jnp.float32))
                    npa = jnp.where(done > 0, jnp.zeros_like(npa), npa)
                    return (nobs, nst, nc, npa, key, ret, done, steps, spread, rng), None
                init = (obs, env_st, carry, prev_action, key,
                        jnp.float32(0.0), jnp.float32(0.0),
                        jnp.float32(0.0), jnp.zeros(2, jnp.float32), key)
                (_, _, _, _, _, ep_ret, ep_done, ep_steps, ep_spread, _), _ = jax.lax.scan(
                    step_fn, init, length=_MAX_STEPS)
                # per-episode mean over the active (pre-clear) steps
                return ep_ret, ep_steps, ep_done, ep_spread / jnp.maximum(ep_steps, 1.0)
            keys = jax.random.split(rng_key, n_episodes)
            rets, steps, dones, spreads = jax.vmap(run_episode)(keys)
            sp = jnp.mean(spreads, axis=0)   # [mean qstd, mean qrange] over actions/episodes
            return jnp.mean(rets), jnp.mean(steps), jnp.mean(dones), sp[0], sp[1]
        return _evaluate

    # ── Phase 1 — Train / load agent ─────────────────────────────────────────

    tag = ("SAC+LD" if args.approximate_lambda else "SAC") + (
        "+GVD" if args.gvd else ""
    )
    print(f"Building {tag} agent for Battleship {args.rows}x{args.cols} "
          f"(memory={args.memory_type}, hidden={args.memory_hidden_dim})…")
    state, fns, debug_fns = _build_agent(args.seed)
    evaluate = _make_evaluate(fns)
    # Critic-greedy eval is discrete-only (needs the all-actions critic).
    evaluate_critic = (
        _make_evaluate_critic(debug_fns)
        if getattr(debug_fns, "predict_qpi", None) is not None else None
    )

    # ── shared probe helpers (used by Phase 2/3 and periodic probe-eval) ───────
    #
    # Factored out so a lightweight probe can be (re)trained and scored every
    # --probe-eval-interval rounds *during* agent training, reusing the same
    # collection / training / scoring code as the final full-quality probe.

    @partial(jax.jit, static_argnames=["n_steps"])
    def collect_rollout(agent_state, key, n_steps):
        key, rk = jax.random.split(key)
        obs, env_st = env.reset(rk, env_params)
        carry = zero_carry()
        prev_action = zero_prev_action()

        def step_fn(scan_carry, _):
            obs, env_st, carry, prev_action, key = scan_carry
            board = env_st.board.reshape(-1)        # true ships
            hits_misses = env_st.hits_misses.reshape(-1)  # shots so far

            key, sk, ek, eps_key = jax.random.split(key, 4)
            raw, new_carry = fns.predict(
                agent_state, obs, carry, sk, deterministic=False,
                prev_action=prev_action,
            )
            policy_action = jnp.round(raw).astype(jnp.int32)
            # epsilon-greedy over *legal* actions (illegal shots are wasted)
            legal = mask_fn(obs)
            random_action = jax.random.categorical(
                eps_key, jnp.where(legal > 0, 0.0, -1e9)
            ).astype(jnp.int32)
            use_random = jax.random.uniform(eps_key) < args.collect_epsilon
            action = jnp.where(use_random, random_action, policy_action)
            # The next prev-action input is the *executed* action's one-hot
            # (handles the epsilon override transparently — encode whatever
            # was executed, not what predict proposed).
            new_prev_action = jax.nn.one_hot(action, N)

            next_obs, next_st, _, done, _ = env.step(
                ek, env_st, action, env_params
            )
            carry_out = jnp.where(done, zero_carry(), new_carry)
            prev_out = jnp.where(done, zero_prev_action(), new_prev_action)
            return (next_obs, next_st, carry_out, prev_out, key), {
                # The carry is the memory state (the probe's input).
                "carries": new_carry[:CARRY_DIM],
                "board_masks": board.astype(jnp.float32),
                "hits_misses": hits_misses.astype(jnp.float32),
                "dones": done.astype(jnp.float32),
            }

        _, data = jax.lax.scan(
            step_fn, (obs, env_st, carry, prev_action, key), length=n_steps
        )
        return data

    def _collect_and_parse(agent_state, seed_offset, n_steps):
        data = collect_rollout(agent_state, jax.random.key(args.seed + seed_offset), n_steps)
        c = np.array(data["carries"])
        b = np.array(data["board_masks"])
        hm = np.array(data["hits_misses"])
        d = np.array(data["dones"])
        di = np.where(d > 0.5)[0]
        es = np.concatenate([[0], di + 1])
        es = es[es < len(c)]
        eb = np.concatenate([es, [len(c)]])
        return c, b, hm, eb

    @partial(jax.jit, static_argnames=["n_steps"])
    def collect_qpi(agent_state, key, n_steps):
        """Deterministic actor-greedy rollout recording, per step, the critic's
        per-action Q and the actor's per-action π (via debug_fns.predict_qpi) —
        the data behind the GT/critic/actor heatmaps."""
        key, rk = jax.random.split(key)
        obs, env_st = env.reset(rk, env_params)
        carry = zero_carry()
        prev_action = zero_prev_action()

        def step_fn(scan_carry, _):
            obs, env_st, carry, prev_action, key = scan_carry
            board = env_st.board.reshape(-1)
            hits_misses = env_st.hits_misses.reshape(-1)
            key, ek = jax.random.split(key)
            q, probs, new_carry = debug_fns.predict_qpi(
                agent_state, obs, carry, prev_action=prev_action)
            legal = mask_fn(obs)
            action = jnp.argmax(jnp.where(legal > 0, probs, -1.0)).astype(jnp.int32)
            new_prev_action = jax.nn.one_hot(action, N)
            next_obs, next_st, _, done, _ = env.step(ek, env_st, action, env_params)
            carry_out = jnp.where(done, zero_carry(), new_carry)
            prev_out = jnp.where(done, zero_prev_action(), new_prev_action)
            return (next_obs, next_st, carry_out, prev_out, key), {
                "board_masks": board.astype(jnp.float32),
                "hits_misses": hits_misses.astype(jnp.float32),
                "q": q.astype(jnp.float32),
                "probs": probs.astype(jnp.float32),
                "dones": done.astype(jnp.float32),
            }

        _, data = jax.lax.scan(
            step_fn, (obs, env_st, carry, prev_action, key), length=n_steps
        )
        return data

    def _collect_qpi_episode(agent_state, seed_offset, n_steps):
        """Roll the agent and return the LONGEST complete episode's
        (board, hits_misses, q, probs) for the heatmap figures."""
        data = collect_qpi(agent_state, jax.random.key(args.seed + seed_offset), n_steps)
        b = np.array(data["board_masks"]); hm = np.array(data["hits_misses"])
        q = np.array(data["q"]); probs = np.array(data["probs"])
        d = np.array(data["dones"])
        di = np.where(d > 0.5)[0]
        es = np.concatenate([[0], di + 1]); es = es[es < len(b)]
        eb = np.concatenate([es, [len(b)]])
        lens = np.diff(eb)
        if len(lens) == 0:
            return b, hm, q, probs
        best = int(np.argmax(lens)); s0, e0 = int(eb[best]), int(eb[best + 1])
        return b[s0:e0], hm[s0:e0], q[s0:e0], probs[s0:e0]

    def _targets_from(board_masks, hits_misses):
        # [ship(N) | fired(N)] — the probe's two heads.
        ship_t = jnp.array(board_masks)
        fired_t = (jnp.array(hits_misses) != 0).astype(jnp.float32)
        return jnp.concatenate([ship_t, fired_t], axis=-1)

    # One optimizer / jitted SGD chunk, reused across every probe we train
    # (final + each periodic eval), so the chunk compiles only once.
    _probe_opt = optax.adam(args.probe_lr)

    @partial(jax.jit, static_argnames=["n_steps", "batch_size"])
    def _probe_train_chunk(params, opt_state, key, c_data, t_data, n_steps, batch_size):
        n = c_data.shape[0]
        def body(carry, _):
            params, opt_state, key = carry
            key, bk = jax.random.split(key)
            idx = jax.random.randint(bk, (batch_size,), 0, n)
            def loss_fn(p):
                logits = probe_forward(p, c_data[idx])
                return optax.sigmoid_binary_cross_entropy(logits, t_data[idx]).mean()
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_os = _probe_opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_os, key), loss
        (params, opt_state, key), losses = jax.lax.scan(body, (params, opt_state, key), length=n_steps)
        return params, opt_state, key, losses[-1]

    def _train_probe(c_jnp, t_jnp, init_key, train_key, n_steps, verbose=False):
        # Probe has 2N outputs: first N = P(ship), last N = P(cell was fired at).
        # The fired head tests whether the memory retains *which* cells it shot.
        params = init_probe_params(init_key, CARRY_DIM, 2 * N, args.probe_hidden_dim)
        opt_state = _probe_opt.init(params)
        _CHUNK = 50_000
        n_chunks = n_steps // _CHUNK
        remainder = n_steps % _CHUNK
        key = train_key
        steps_done = 0

        def _step(nsteps):
            nonlocal params, opt_state, key, steps_done
            params, opt_state, key, loss_val = _probe_train_chunk(
                params, opt_state, key, c_jnp, t_jnp, nsteps, args.probe_batch_size
            )
            steps_done += nsteps
            if verbose:
                print(f"  step {steps_done:7d}  bce={float(loss_val):.6f}")
                if _wandb is not None:
                    _wandb.log({"probe_step": steps_done, "probe/bce": float(loss_val)})

        for _ in (tqdm(range(n_chunks), desc="Probe") if verbose else range(n_chunks)):
            _step(_CHUNK)
        if remainder > 0:
            _step(remainder)
        return params

    def _probe_metrics_from_probs(probs_all, t_boards, t_hm):
        """Headline decodability metrics from precomputed host probabilities.

        Split out from :func:`_probe_metrics` so the multi-seed path can score
        from predictions already pulled to host (pure numpy — no GPU op), which
        is what lets the matplotlib rendering overlap the next training round.
        """
        probs_all = np.asarray(probs_all)
        probs = probs_all[:, :N]            # P(ship)
        fired_probs = probs_all[:, N:]      # P(fired)
        preds = (probs > 0.5).astype(np.float32)
        targets = np.asarray(t_boards).astype(np.float32)
        hm = np.asarray(t_hm)
        fired_targets = (hm != 0).astype(np.float32)
        correct = (preds == targets).astype(np.float32)
        fired = hm != 0          # observed cells (retention)
        unfired = ~fired         # hidden cells (inference)

        def _acc(mask):
            m = mask.astype(np.float32)
            return float((correct * m).sum() / max(m.sum(), 1.0))

        def _balanced(mask):
            s = _acc(mask & (targets == 1))
            w = _acc(mask & (targets == 0))
            return 0.5 * (s + w)

        fired_pred = (fired_probs > 0.5).astype(np.float32)
        return {
            "overall_acc": float(correct.mean()),
            "fired_auroc": _auroc(probs[fired], targets[fired]),
            "fired_balanced": _balanced(fired),
            "fired_acc": _acc(fired),
            "unfired_auroc": _auroc(probs[unfired], targets[unfired]),
            "unfired_balanced": _balanced(unfired),
            "unfired_acc": _acc(unfired),
            "frac_fired": float(fired.mean()),
            "fired_pred_auroc": _auroc(fired_probs, fired_targets),
            "fired_pred_acc": float((fired_pred == fired_targets).mean()),
        }

    def _probe_metrics(probe_params, t_carries, t_boards, t_hm):
        """Headline decodability metrics on a held-out test set (no plots)."""
        probs_all = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(t_carries))))
        return _probe_metrics_from_probs(probs_all, t_boards, t_hm)

    def _run_probe_eval(agent_state, rnd):
        """Collect → train a lightweight probe → score it; log under probe_eval/.

        Uses the *current* agent state, so probe_eval/* curves show how the
        memory's board decodability grows over agent training.  Cheaper than the
        final probe (--probe-eval-steps / --probe-eval-collect-steps) and trained
        fresh each time (the memory it decodes has changed).
        """
        c, b, hm, _eb = _collect_and_parse(agent_state, 5000 + rnd, args.probe_eval_collect_steps)
        tc, tb, thm, _teb = _collect_and_parse(agent_state, 6000 + rnd, args.probe_eval_collect_steps)
        params = _train_probe(
            jnp.array(c), _targets_from(b, hm),
            jax.random.key(args.seed + 40000 + rnd),
            jax.random.key(args.seed + 50000 + rnd),
            args.probe_eval_steps,
        )
        m = _probe_metrics(params, tc, tb, thm)
        step = rnd * args.train_steps
        print(f"  [probe-eval] fired AUROC={m['fired_auroc']:.3f}  "
              f"unfired AUROC={m['unfired_auroc']:.3f}  overall={m['overall_acc']:.1%}")
        if _wandb is not None:
            _wandb.log({"env_interactions": step,
                        **{f"probe_eval/{k}": v for k, v in m.items()}})

        # Per-eval visuals (rendered from the held-out eval test set; round-stamped
        # so the on-disk files form a progression and W&B gets a step-sliderable
        # media timeline).  Skipped with --no-probe-eval-vis.
        if args.probe_eval_vis:
            vtag = f"{tag} @ {step} steps"
            vdir = os.path.join(args.output_dir, "probe_eval")
            os.makedirs(vdir, exist_ok=True)
            imgs = {}
            if len(_teb) - 1 >= 1:
                imgs["probe_eval/episode"] = _fig_episode(
                    params, tc, tb, thm, _teb, 0,
                    os.path.join(vdir, f"episode_r{rnd:04d}.png"), vtag, args.vis_frames)
            imgs["probe_eval/per_cell_accuracy"] = _fig_accuracy(
                params, tc, tb, thm,
                os.path.join(vdir, f"accuracy_r{rnd:04d}.png"), vtag)
            ret = _fig_retention(
                params, tc, tb, thm, _teb,
                os.path.join(vdir, f"retention_r{rnd:04d}.png"), vtag)
            if ret is not None:
                imgs["probe_eval/retention"] = ret
            mov = None
            lens = np.diff(_teb)
            if len(lens):
                best = int(np.argmax(lens))
                s0, e0 = int(_teb[best]), int(_teb[best + 1])
                mov = _fig_movie(
                    params, tc[s0:e0], tb[s0:e0], thm[s0:e0],
                    os.path.join(vdir, f"movie_r{rnd:04d}"), vtag,
                    "epsilon-soft (collected)")
            if _wandb is not None:
                payload = {"env_interactions": step}
                payload.update({k: _wandb.Image(p) for k, p in imgs.items()})
                if mov is not None:
                    payload["probe_eval/movie"] = _wandb.Video(mov)
                _wandb.log(payload)

    def _log_qpi_heatmaps(agent_state, rnd):
        """Critic-Q / actor-π heatmaps along an actor-greedy rollout (GT/critic/
        actor stacked image + GT|critic|actor video).  Needs only predict_qpi —
        independent of the board probe — so it runs even in the memoryless
        (identity-memory) diagnostic.  Logged under probe_eval/critic_actor*."""
        if getattr(debug_fns, "predict_qpi", None) is None or not args.probe_eval_vis:
            return
        step = rnd * args.train_steps
        vtag = f"{tag} @ {step} steps"
        vdir = os.path.join(args.output_dir, "probe_eval")
        os.makedirs(vdir, exist_ok=True)
        qb, qhm, qq, qp = _collect_qpi_episode(
            agent_state, 7000 + rnd, args.probe_eval_collect_steps)
        qimg = _fig_qpi_episode(
            qb, qhm, qq, qp,
            os.path.join(vdir, f"qpi_r{rnd:04d}.png"), vtag, args.vis_frames)
        qpi_mov = _fig_qpi_movie(
            qb, qhm, qq, qp, os.path.join(vdir, f"qpi_movie_r{rnd:04d}"), vtag)
        if _wandb is not None:
            payload = {"env_interactions": step}
            if qimg is not None:
                payload["probe_eval/critic_actor"] = _wandb.Image(qimg)
            if qpi_mov is not None:
                payload["probe_eval/critic_actor_movie"] = _wandb.Video(qpi_mov)
            _wandb.log(payload)

    # ════════════════════════════════════════════════════════════════════════
    #  Multi-seed pipeline (--num-seeds > 1): train concurrent seeds in one
    #  vmapped kernel, run the full pipeline per seed, and log per-seed (seed_i/)
    #  plus aggregated (agg/ mean,std,sterr) metrics.  Self-contained: exits
    #  before the single-seed module-level code below.
    # ════════════════════════════════════════════════════════════════════════

    if args.num_seeds > 1:
        CONCURRENT = args.concurrent_seeds or args.num_seeds
        if CONCURRENT < 1:
            sys.exit("--concurrent-seeds must be >= 1")
        if args.num_seeds % CONCURRENT != 0:
            sys.exit(
                f"--num-seeds ({args.num_seeds}) must be divisible by "
                f"--concurrent-seeds ({CONCURRENT})."
            )
        seeds = [args.seed + i for i in range(args.num_seeds)]
        n_groups = args.num_seeds // CONCURRENT
        print(
            f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT} "
            f"trained concurrently (vmap); {args.rounds} rounds × "
            f"{args.train_steps} steps each."
        )

        # ── vmapped train / reset / eval over the leading seed axis ───────────
        # fns.train has host-side control flow (auto-prefill) and is NOT
        # vmappable; use the documented vmap-safe split instead — prefill the
        # buffer once, then run the jittable fns.train_unrolled each round,
        # threading a per-round zero env_carry (cf. battleship_sac_mc.py:881-895).
        PREFILL_STEPS = hp.online_batch_size * (
            hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
        )
        _reset_v = jax.jit(jax.vmap(lambda k: env.reset(k, env_params)))
        _prefill_v = jax.jit(
            jax.vmap(
                lambda s, es, k: fns.prefill_buffer(
                    s, env, env_params, es, PREFILL_STEPS, k),
                in_axes=(0, 0, 0),
            ),
            donate_argnums=(0, 1),
        )
        _train_v = jax.jit(
            jax.vmap(
                lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
                in_axes=(0, 0, 0, 0),
            ),
            donate_argnums=(0, 1),
        )

        def _evaluate_v(states_b, keys, n):
            return jax.vmap(lambda s, k: evaluate(s, k, n_episodes=n))(states_b, keys)

        def _evaluate_critic_v(states_b, keys, n):
            return jax.vmap(
                lambda s, k: evaluate_critic(s, k, n_episodes=n))(states_b, keys)

        # ── vmapped collect + probe-train (used by periodic eval and Phase 2/3)
        def _collect_v(states_b, keys, n_steps):
            return jax.vmap(lambda s, k: collect_rollout(s, k, n_steps))(states_b, keys)

        def _ep_bounds(dones, length):
            di = np.where(np.asarray(dones) > 0.5)[0]
            es = np.concatenate([[0], di + 1])
            es = es[es < length]
            return np.concatenate([es, [length]])

        def _targets_b(boards_b, hm_b):
            # [ship(N) | fired(N)] per seed → (S, n, 2N).
            return jnp.concatenate(
                [jnp.asarray(boards_b),
                 (jnp.asarray(hm_b) != 0).astype(jnp.float32)], axis=-1)

        def _train_probe_v(c_b, t_b, init_keys, train_keys, n_steps):
            """Vmapped probe training: (S, n, CARRY_DIM)/(S, n, 2N) → batched params."""
            params = jax.vmap(
                lambda k: init_probe_params(k, CARRY_DIM, 2 * N, args.probe_hidden_dim)
            )(init_keys)
            opt_state = jax.vmap(_probe_opt.init)(params)
            _CHUNK = 50_000
            n_chunks = n_steps // _CHUNK
            remainder = n_steps % _CHUNK
            keys = train_keys

            def _chunk_v(p, o, k, nsteps):
                # vmap over params/opt/key AND the per-seed data slices (c_b/t_b
                # are (S, n, …) — each lane must train on its own slice).
                return jax.vmap(
                    lambda p, o, k, c, t: _probe_train_chunk(
                        p, o, k, c, t, nsteps, args.probe_batch_size),
                    in_axes=(0, 0, 0, 0, 0),
                )(p, o, k, c_b, t_b)

            for _ in range(n_chunks):
                params, opt_state, keys, _loss = _chunk_v(params, opt_state, keys, _CHUNK)
            if remainder > 0:
                params, opt_state, keys, _loss = _chunk_v(
                    params, opt_state, keys, remainder)
            return params

        # ── periodic probe-eval across all seeds, split GPU-compute / CPU-render
        #
        # Async-dispatch overlap: the GPU phase (collect + train probe) pulls all
        # predictions + metrics to host; the caller then dispatches the next
        # training round (async, donating the agent state) and only AFTER that
        # runs the pure-CPU matplotlib rendering, which therefore overlaps the
        # next round's GPU compute.  The _fig_* helpers take probs_all=<host
        # predictions> so rendering fires no GPU op.
        _probe_forward_v = jax.vmap(probe_forward)

        def _probe_eval_compute(states_b, gidxs, rnd):
            ne = args.probe_eval_collect_steps
            ck_tr = jnp.stack([jax.random.key(args.seed + 500000 + rnd + gi) for gi in gidxs])
            ck_te = jnp.stack([jax.random.key(args.seed + 600000 + rnd + gi) for gi in gidxs])
            tr = _collect_v(states_b, ck_tr, ne)
            te = _collect_v(states_b, ck_te, ne)
            c_tr = jnp.asarray(tr["carries"])
            t_tr = _targets_b(tr["board_masks"], tr["hits_misses"])
            ik = jnp.stack([jax.random.key(args.seed + 40000 + rnd + gi) for gi in gidxs])
            tk = jnp.stack([jax.random.key(args.seed + 50000 + rnd + gi) for gi in gidxs])
            params_b = _train_probe_v(c_tr, t_tr, ik, tk, args.probe_eval_steps)
            te_c = jnp.asarray(te["carries"])
            # Pull predictions (and test data) to host — this block_until_ready is
            # the only sync point; everything downstream is pure CPU.
            preds = np.array(jax.nn.sigmoid(_probe_forward_v(params_b, te_c)))  # (S, n, 2N)
            te_c = np.array(te_c); te_b = np.array(te["board_masks"])
            te_hm = np.array(te["hits_misses"]); te_d = np.array(te["dones"])
            per_seed = [
                _probe_metrics_from_probs(preds[j], te_b[j], te_hm[j])
                for j in range(len(gidxs))
            ]
            return {"gidxs": list(gidxs), "rnd": rnd, "preds": preds, "te_c": te_c,
                    "te_b": te_b, "te_hm": te_hm, "te_d": te_d, "per_seed": per_seed}

        def _probe_eval_render(host):
            """Pure-CPU: log aggregated/per-seed metrics + render every seed's
            figures (overlaps the next training round under async dispatch)."""
            gidxs = host["gidxs"]; rnd = host["rnd"]; step = rnd * args.train_steps
            per_seed = host["per_seed"]
            payload = {"env_interactions": step}
            for k in per_seed[0]:
                payload.update(_agg([m[k] for m in per_seed], f"probe_eval/agg/{k}"))
                for j, gi in enumerate(gidxs):
                    payload[f"seed_{gi}/probe_eval/{k}"] = per_seed[j][k]
            print(f"  [probe-eval] agg fired AUROC={payload['probe_eval/agg/fired_auroc/mean']:.3f}"
                  f"±{payload['probe_eval/agg/fired_auroc/sterr']:.3f}  "
                  f"unfired AUROC={payload['probe_eval/agg/unfired_auroc/mean']:.3f}")
            if _wandb is not None:
                _wandb.log(payload)
            if not args.probe_eval_vis:
                return
            img_payload = {"env_interactions": step}
            for j, gi in enumerate(gidxs):
                vtag = f"{tag} @ {step} steps (seed {gi})"
                vdir = os.path.join(args.output_dir, f"seed{gi}", "probe_eval")
                os.makedirs(vdir, exist_ok=True)
                pj, tc, tb, thm = host["preds"][j], host["te_c"][j], host["te_b"][j], host["te_hm"][j]
                eb = _ep_bounds(host["te_d"][j], len(tc))
                if len(eb) - 1 >= 1:
                    ep = _fig_episode(None, tc, tb, thm, eb, 0,
                                      os.path.join(vdir, f"episode_r{rnd:04d}.png"),
                                      vtag, args.vis_frames, probs_all=pj)
                    if _wandb is not None:
                        img_payload[f"seed_{gi}/probe_eval/episode"] = _wandb.Image(ep)
                acc = _fig_accuracy(None, tc, tb, thm,
                                    os.path.join(vdir, f"accuracy_r{rnd:04d}.png"),
                                    vtag, probs_all=pj)
                ret = _fig_retention(None, tc, tb, thm, eb,
                                     os.path.join(vdir, f"retention_r{rnd:04d}.png"),
                                     vtag, probs_all=pj)
                if _wandb is not None:
                    img_payload[f"seed_{gi}/probe_eval/per_cell_accuracy"] = _wandb.Image(acc)
                    if ret is not None:
                        img_payload[f"seed_{gi}/probe_eval/retention"] = _wandb.Image(ret)
            if _wandb is not None and len(img_payload) > 1:
                _wandb.log(img_payload)

        def _qpi_collect_host(state0, gi0, rnd):
            """GPU: collect one actor-greedy episode's Q/π for seed gi0 → host."""
            if getattr(debug_fns, "predict_qpi", None) is None or not args.probe_eval_vis:
                return None
            qb, qhm, qq, qp = _collect_qpi_episode(state0, 7000 + rnd, args.probe_eval_collect_steps)
            return (qb, qhm, qq, qp, gi0, rnd)

        def _qpi_render_host(h):
            if h is None:
                return
            qb, qhm, qq, qp, gi0, rnd = h
            step = rnd * args.train_steps
            vtag = f"{tag} @ {step} steps (seed {gi0})"
            vdir = os.path.join(args.output_dir, f"seed{gi0}", "probe_eval")
            os.makedirs(vdir, exist_ok=True)
            qimg = _fig_qpi_episode(qb, qhm, qq, qp,
                                    os.path.join(vdir, f"qpi_r{rnd:04d}.png"), vtag, args.vis_frames)
            qmov = _fig_qpi_movie(qb, qhm, qq, qp,
                                  os.path.join(vdir, f"qpi_movie_r{rnd:04d}"), vtag)
            if _wandb is not None:
                payload = {"env_interactions": step}
                if qimg is not None:
                    payload[f"seed_{gi0}/probe_eval/critic_actor"] = _wandb.Image(qimg)
                if qmov is not None:
                    payload[f"seed_{gi0}/probe_eval/critic_actor_movie"] = _wandb.Video(qmov)
                _wandb.log(payload)

        # ── per-round history (across all groups) for the smoothed final metric
        return_hist = {gi: [] for gi in range(args.num_seeds)}
        steps_hist = {gi: [] for gi in range(args.num_seeds)}

        def run_group(group_idx, gidxs):
            svals = [seeds[gi] for gi in gidxs]
            print(f"\n{'=' * 60}\nGroup {group_idx + 1}/{n_groups}  "
                  f"seeds={svals} (idx {gidxs[0]}–{gidxs[-1]})\n{'=' * 60}")
            states = [
                state if gi == 0 else _build_agent(seeds[gi])[0] for gi in gidxs
            ]
            batched = _stack_states(states)
            keys = jnp.stack([jax.random.key(sv) for sv in svals])
            keys, reset_keys = _split_each(keys)
            _obs, env_state = _reset_v(reset_keys)
            keys, prefill_keys = _split_each(keys)
            print(f"  prefilling {PREFILL_STEPS} steps/seed…")
            batched, env_state = _prefill_v(batched, env_state, prefill_keys)
            # Fresh zero env_carry each round (matches fns.train's per-call reset);
            # reused across rounds, so NOT donated.
            zero_carry_b = jnp.zeros((len(gidxs), AGENT_CARRY_DIM), dtype=jnp.float32)

            # Prefetch round 1's training (async); each iteration renders the
            # current round's figures while the NEXT round trains on the GPU.
            keys, train_keys = _split_each(keys)
            pending = _train_v(batched, env_state, zero_carry_b, train_keys)

            for rnd in tqdm(range(1, args.rounds + 1), desc=f"Group {group_idx + 1}"):
                batched, env_state, _ec, metrics = pending
                keys, eval_keys = _split_each(keys)
                returns, steps, cleared = _evaluate_v(batched, eval_keys, 10)
                returns = np.array(returns); steps = np.array(steps); cleared = np.array(cleared)
                cg = None
                if evaluate_critic is not None:
                    keys, cg_keys = _split_each(keys)
                    cgr, cgs, cgc, qsp, qrg = _evaluate_critic_v(batched, cg_keys, 10)
                    cg = (np.array(cgr), np.array(cgs), np.array(cgc),
                          np.array(qsp), np.array(qrg))

                # GPU work that still reads `batched` (probe-eval collect/train,
                # qpi collect) → pulled to host BEFORE the next round is dispatched
                # (which donates `batched`).
                do_probe = args.probe_eval_interval > 0 and rnd % args.probe_eval_interval == 0
                do_qpi = _qpi_interval > 0 and rnd % _qpi_interval == 0
                probe_host = _probe_eval_compute(batched, gidxs, rnd) if do_probe else None
                qpi_host = (_qpi_collect_host(_unstack_state(batched, 0), gidxs[0], rnd)
                            if do_qpi else None)

                # All reads of batched/env_state are now on host — dispatch the
                # NEXT round's training (async, donates them).  Its GPU compute
                # overlaps the matplotlib rendering at the end of this iteration.
                if rnd < args.rounds:
                    keys, train_keys = _split_each(keys)
                    pending = _train_v(batched, env_state, zero_carry_b, train_keys)

                for j, gi in enumerate(gidxs):
                    return_hist[gi].append(float(returns[j]))
                    steps_hist[gi].append(float(steps[j]))

                step = rnd * args.train_steps
                print(f"Round {rnd:4d}/{args.rounds}  "
                      f"return={float(returns.mean()):7.1f}±{float(returns.std()):.1f}  "
                      f"steps_to_clear={float(steps.mean()):5.1f}  "
                      f"cleared={float(cleared.mean()):.2f}")
                if _wandb is not None:
                    payload = {"round": rnd, "env_interactions": step}
                    payload.update(_agg(returns, "agg/return"))
                    payload.update(_agg(steps, "agg/steps_to_clear"))
                    payload.update(_agg(cleared, "agg/cleared_frac"))
                    if cg is not None:
                        payload.update(_agg(cg[0], "agg/critic_greedy_return"))
                        payload.update(_agg(cg[1], "agg/critic_greedy_steps_to_clear"))
                        payload.update(_agg(cg[2], "agg/critic_greedy_cleared_frac"))
                        payload.update(_agg(cg[3], "agg/q_action_spread"))
                        payload.update(_agg(cg[4], "agg/q_action_range"))
                    for mk, mv in metrics.items():
                        payload.update(_agg(np.array(mv), f"agg/{mk}"))
                    for j, gi in enumerate(gidxs):
                        payload[f"seed_{gi}/agent/mean_return"] = float(returns[j])
                        payload[f"seed_{gi}/agent/steps_to_clear"] = float(steps[j])
                        payload[f"seed_{gi}/agent/cleared_frac"] = float(cleared[j])
                        if cg is not None:
                            payload[f"seed_{gi}/agent/critic_greedy_steps_to_clear"] = float(cg[1][j])
                            payload[f"seed_{gi}/agent/q_action_spread"] = float(cg[3][j])
                            payload[f"seed_{gi}/agent/q_action_range"] = float(cg[4][j])
                        for mk, mv in metrics.items():
                            payload[f"seed_{gi}/agent/{mk}"] = float(np.array(mv)[j])
                    _wandb.log(payload)

                # CPU rendering — overlaps the dispatched round R+1 GPU training.
                if probe_host is not None:
                    _probe_eval_render(probe_host)
                if qpi_host is not None:
                    _qpi_render_host(qpi_host)
            return batched

        # ── train each group (or load saved per-seed agents) ──────────────────
        indexed = list(range(args.num_seeds))
        groups = [indexed[g * CONCURRENT:(g + 1) * CONCURRENT] for g in range(n_groups)]
        seed_states = [None] * args.num_seeds
        if not args.skip_train:
            for g, gidxs in enumerate(groups):
                batched = run_group(g, gidxs)
                for j, gi in enumerate(gidxs):
                    seed_states[gi] = _unstack_state(batched, j)
                    p = os.path.join(args.output_dir, f"agent_seed{gi}.pkl")
                    leaves, treedef = jax.tree.flatten(seed_states[gi])
                    #with open(p, "wb") as f:
                    #    pickle.dump({"leaves": [np.array(l) for l in leaves],
                    #                 "treedef": treedef}, f)
            print(f"Saved {args.num_seeds} per-seed agents → {args.output_dir}/agent_seed*.pkl")
        else:
            for gi in range(args.num_seeds):
                p = os.path.join(args.output_dir, f"agent_seed{gi}.pkl")
                print(f"Loading agent ← {p}")
                with open(p, "rb") as f:
                    saved = pickle.load(f)
                _, treedef = jax.tree.flatten(state)
                seed_states[gi] = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

        # ── smoothed final return / steps (per seed, then aggregated) ─────────
        if not args.skip_train:
            W = max(1, args.final_return_window)
            fin_ret = np.array([np.mean(return_hist[gi][-W:]) for gi in range(args.num_seeds)])
            fin_steps = np.array([np.mean(steps_hist[gi][-W:]) for gi in range(args.num_seeds)])
            # 20-episode eval of the final policy, aggregated across seeds.
            all_states = _stack_states(seed_states)
            fkeys = jnp.stack([jax.random.key(seeds[gi] + 999) for gi in range(args.num_seeds)])
            fr, _fs, _fc = _evaluate_v(all_states, fkeys, 20)
            fr = np.array(fr)
            agg = {}
            agg.update(_agg(fin_ret, "final/return_smoothed"))
            agg.update(_agg(fin_steps, "final/steps_to_clear_smoothed"))
            agg.update(_agg(fr, "final/return"))
            print(f"\n{'=' * 60}\nAggregated over {args.num_seeds} seed(s) "
                  f"(smoothed over final {W} round(s)):\n"
                  f"  final/return_smoothed = {agg['final/return_smoothed/mean']:.1f} "
                  f"± {agg['final/return_smoothed/sterr']:.1f} (sterr)\n"
                  f"  final/steps_to_clear_smoothed = "
                  f"{agg['final/steps_to_clear_smoothed/mean']:.1f} "
                  f"± {agg['final/steps_to_clear_smoothed/sterr']:.1f}\n{'=' * 60}")
            if _wandb is not None:
                _wandb.log(agg)
                _wandb.summary.update(agg)

        if _memoryless:
            print("[memoryless diagnostic] done (skipping collect/probe/vis phases).")
            if _wandb is not None:
                _wandb.finish()
            sys.exit(0)

        # ── Phase 2/3 — collect + train probe, per seed (vmapped) ─────────────
        all_states = _stack_states(seed_states)
        if not args.skip_collect:
            print(f"Collecting {args.collect_steps} train+test steps per seed (vmapped)…")
            ck_tr = jnp.stack([jax.random.key(seeds[gi] + 1000) for gi in range(args.num_seeds)])
            ck_te = jnp.stack([jax.random.key(seeds[gi] + 2000) for gi in range(args.num_seeds)])
            tr = _collect_v(all_states, ck_tr, args.collect_steps)
            te = _collect_v(all_states, ck_te, args.collect_steps)
            tr_c = np.array(tr["carries"]); tr_b = np.array(tr["board_masks"])
            tr_hm = np.array(tr["hits_misses"]); tr_d = np.array(tr["dones"])
            te_c = np.array(te["carries"]); te_b = np.array(te["board_masks"])
            te_hm = np.array(te["hits_misses"]); te_d = np.array(te["dones"])
            for gi in range(args.num_seeds):
                for nm, c, b, hm, d in (
                    (f"dataset_seed{gi}.pkl", tr_c[gi], tr_b[gi], tr_hm[gi], tr_d[gi]),
                    (f"test_dataset_seed{gi}.pkl", te_c[gi], te_b[gi], te_hm[gi], te_d[gi]),
                ):
                    eb = _ep_bounds(d, len(c))
                    #with open(os.path.join(args.output_dir, nm), "wb") as f:
                    #    pickle.dump({"carries": c, "board_masks": b, "hits_misses": hm,
                    #                 "ep_bounds": eb, "rows": args.rows, "cols": args.cols,
                    #                 "tag": tag}, f)
            print(f"  → {args.output_dir}/dataset_seed*.pkl, test_dataset_seed*.pkl")
        else:
            tr_c = tr_b = tr_hm = None
            te_c, te_b, te_hm, te_d = [], [], [], []
            tr_list = []
            for gi in range(args.num_seeds):
                with open(os.path.join(args.output_dir, f"dataset_seed{gi}.pkl"), "rb") as f:
                    ds = pickle.load(f)
                tr_list.append((ds["carries"], ds["board_masks"], ds["hits_misses"]))
                with open(os.path.join(args.output_dir, f"test_dataset_seed{gi}.pkl"), "rb") as f:
                    tds = pickle.load(f)
                te_c.append(tds["carries"]); te_b.append(tds["board_masks"])
                te_hm.append(tds["hits_misses"]); te_d.append(tds["ep_bounds"])
            tr_c = np.stack([x[0] for x in tr_list]); tr_b = np.stack([x[1] for x in tr_list])
            tr_hm = np.stack([x[2] for x in tr_list])
            te_c = np.stack(te_c); te_b = np.stack(te_b); te_hm = np.stack(te_hm)
            te_d = None  # ep_bounds reloaded per seed below when needed

        if not args.skip_probe:
            print(f"Training probe per seed ({args.probe_steps} steps, vmapped)…")
            c_b = jnp.asarray(tr_c)
            t_b = _targets_b(tr_b, tr_hm)
            ik = jnp.stack([jax.random.key(seeds[gi] + 20000) for gi in range(args.num_seeds)])
            tk = jnp.stack([jax.random.key(seeds[gi] + 30000) for gi in range(args.num_seeds)])
            probe_b = _train_probe_v(c_b, t_b, ik, tk, args.probe_steps)
            for gi in range(args.num_seeds):
                pp = _unstack_state(probe_b, gi)
                leaves, td = jax.tree.flatten(pp)
                #with open(os.path.join(args.output_dir, f"probe_seed{gi}.pkl"), "wb") as f:
                #    pickle.dump({"leaves": [np.array(l) for l in leaves], "treedef": td}, f)
            print(f"  → {args.output_dir}/probe_seed*.pkl")
        else:
            params_list = []
            ref = init_probe_params(jax.random.key(0), CARRY_DIM, 2 * N, args.probe_hidden_dim)
            _, td = jax.tree.flatten(ref)
            for gi in range(args.num_seeds):
                with open(os.path.join(args.output_dir, f"probe_seed{gi}.pkl"), "rb") as f:
                    saved = pickle.load(f)
                params_list.append(td.unflatten([jnp.array(l) for l in saved["leaves"]]))
            probe_b = _stack_states(params_list)

        # ── Phase 4 — final metrics (aggregated) + per-seed visualisation ─────
        print("Computing final probe metrics per seed (held-out test set)…")
        per_seed = [
            _probe_metrics(_unstack_state(probe_b, gi),
                           np.asarray(te_c[gi]), np.asarray(te_b[gi]), np.asarray(te_hm[gi]))
            for gi in range(args.num_seeds)
        ]
        final_payload = {}
        for k in per_seed[0]:
            final_payload.update(_agg([m[k] for m in per_seed], f"eval/agg/{k}"))
            for gi in range(args.num_seeds):
                final_payload[f"eval/seed_{gi}/{k}"] = per_seed[gi][k]
        print(f"  eval/agg/fired_auroc = {final_payload['eval/agg/fired_auroc/mean']:.3f} "
              f"± {final_payload['eval/agg/fired_auroc/sterr']:.3f} (sterr)")
        print(f"  eval/agg/unfired_auroc = {final_payload['eval/agg/unfired_auroc/mean']:.3f} "
              f"± {final_payload['eval/agg/unfired_auroc/sterr']:.3f}")
        if _wandb is not None:
            _wandb.log(final_payload)
            _wandb.summary.update(final_payload)

        print("Rendering per-seed visualisations…")
        for gi in range(args.num_seeds):
            vdir = os.path.join(args.output_dir, f"seed{gi}")
            os.makedirs(vdir, exist_ok=True)
            pp = _unstack_state(probe_b, gi)
            tc, tb, thm = np.asarray(te_c[gi]), np.asarray(te_b[gi]), np.asarray(te_hm[gi])
            if not args.skip_collect:
                eb = _ep_bounds(te_d[gi], len(tc))
            else:
                with open(os.path.join(args.output_dir, f"test_dataset_seed{gi}.pkl"), "rb") as f:
                    eb = pickle.load(f)["ep_bounds"]
            vtag = f"{tag} (seed {gi})"
            n_vis = min(args.vis_episodes, len(eb) - 1)
            for vi in range(n_vis):
                _fig_episode(pp, tc, tb, thm, eb, vi,
                             os.path.join(vdir, f"board_probe_ep{vi + 1}.png"), vtag, args.vis_frames)
            _fig_accuracy(pp, tc, tb, thm,
                          os.path.join(vdir, "board_probe_accuracy.png"), vtag)
            _fig_retention(pp, tc, tb, thm, eb,
                           os.path.join(vdir, "board_probe_retention.png"), vtag)
            if args.mp4:
                lens = np.diff(eb)
                if len(lens):
                    best = int(np.argmax(lens))
                    s0, e0 = int(eb[best]), int(eb[best + 1])
                    _fig_movie(pp, tc[s0:e0], tb[s0:e0], thm[s0:e0],
                               os.path.join(vdir, "board_probe"), vtag,
                               "epsilon-soft (collected)")
        print(f"  → per-seed images under {args.output_dir}/seed*/")
        if _wandb is not None:
            _wandb.finish()
        print("Done (multi-seed).")
        sys.exit(0)

    if not args.skip_train:
        if args.resume_from:
            print(f"Resuming training ← {args.resume_from}")
            with open(args.resume_from, "rb") as f:
                _ck = pickle.load(f)
            # Transplant the checkpointed state into the freshly-built `fns`
            # (built above from the CURRENT CLI hyperparameters).  Net shapes
            # are unchanged by tau/lr/gvd_coef, so the pytree structure matches.
            state = _ck["state_treedef"].unflatten(
                [jnp.array(l) for l in _ck["state_leaves"]])
            env_state = _ck["env_treedef"].unflatten(
                [jnp.array(l) for l in _ck["env_leaves"]])
            key = jax.random.wrap_key_data(jnp.asarray(_ck["key"]))
            _start_round = int(_ck["round"])
            print(f"  resumed at round {_start_round}; continuing with "
                  f"tau={args.tau} critic_lr={args.critic_lr} fe_lr={args.fe_lr} "
                  f"gvd_sf_lr={args.gvd_sf_lr} gvd_coef={args.gvd_coef}")
        else:
            key = jax.random.key(args.seed)
            key, reset_key = jax.random.split(key)
            _, env_state = env.reset(reset_key, env_params)
            _start_round = 0
        total = args.rounds * args.train_steps
        print(f"Training for {args.rounds} × {args.train_steps} = {total} steps…")

        for rnd in tqdm(range(_start_round + 1, args.rounds + 1), desc="Training"):
            # Linear alpha anneal (fixed schedule, autotune off): decay from
            # args.alpha at round 1 to args.alpha_anneal_final at the last round.
            # Override the carried state's alpha/log_alpha before this round's
            # train() so update_step uses the scheduled value.
            cur_alpha = args.alpha
            if args.alpha_anneal_final is not None:
                frac = (rnd - 1) / max(1, args.rounds - 1)
                cur_alpha = float(args.alpha + (args.alpha_anneal_final - args.alpha) * min(1.0, frac))
                state = state._replace(
                    alpha=jnp.asarray(cur_alpha, dtype=jnp.float32),
                    log_alpha=jnp.asarray(np.log(cur_alpha), dtype=jnp.float32),
                )
            key, train_key = jax.random.split(key)
            state, env_state, metrics = fns.train(
                state, env, env_params, env_state, train_key
            )
            key, eval_key = jax.random.split(key)
            _mr, _steps, _cleared = evaluate(state, eval_key, n_episodes=10)
            mr, steps_to_clear, cleared = float(_mr), float(_steps), float(_cleared)
            # Critic-greedy probe (argmax_a Q) alongside the actor-greedy eval.
            cg_log = {}
            cg_str = ""
            if evaluate_critic is not None:
                key, cg_key = jax.random.split(key)
                _cgr, _cgs, _cgc, _qspread, _qrange = evaluate_critic(
                    state, cg_key, n_episodes=10)
                cg_log = {
                    "agent/critic_greedy_return": float(_cgr),
                    "agent/critic_greedy_steps_to_clear": float(_cgs),
                    "agent/critic_greedy_cleared_frac": float(_cgc),
                    # flatness of Q across legal actions (≈0 ⇒ no per-action
                    # ranking ⇒ no advantage signal for the actor).
                    "agent/q_action_spread": float(_qspread),
                    "agent/q_action_range": float(_qrange),
                }
                cg_str = (f"  [critic-greedy steps={float(_cgs):5.1f} "
                          f"q_spread={float(_qspread):.4f} q_range={float(_qrange):.4f}]")
            print(
                f"  round {rnd:3d}/{args.rounds}  return={mr:7.1f}  "
                f"steps_to_clear={steps_to_clear:5.1f}  cleared={cleared:.2f}  "
                f"critic_loss={float(metrics.get('critic_loss', jnp.nan)):.4f}{cg_str}"
            )
            if _wandb is not None:
                _wandb.log({
                    "round": rnd,
                    "env_interactions": rnd * args.train_steps,
                    "agent/alpha": cur_alpha,
                    "agent/mean_return": mr,
                    "agent/steps_to_clear": steps_to_clear,
                    "agent/cleared_frac": cleared,
                    **cg_log,
                    **{f"agent/{k}": float(v) for k, v in metrics.items()},
                })

            if args.probe_eval_interval > 0 and rnd % args.probe_eval_interval == 0:
                _run_probe_eval(state, rnd)
            # Critic/actor heatmaps render on their own cadence (probe-independent),
            # so they appear even in the memoryless (identity) diagnostic.
            if _qpi_interval > 0 and rnd % _qpi_interval == 0:
                _log_qpi_heatmaps(state, rnd)

            if args.save_checkpoint_at and rnd == args.save_checkpoint_at:
                _ckpath = os.path.join(args.output_dir, f"train_ckpt_r{rnd}.pkl")
                _sl, _st = jax.tree.flatten(state)
                _el, _et = jax.tree.flatten(env_state)
                with open(_ckpath, "wb") as f:
                    pickle.dump({
                        "state_leaves": [np.array(l) for l in _sl],
                        "state_treedef": _st,
                        "env_leaves": [np.array(l) for l in _el],
                        "env_treedef": _et,
                        "key": np.array(jax.random.key_data(key)),
                        "round": rnd,
                    }, f)
                print(f"\nSaved training checkpoint → {_ckpath} (round {rnd}); exiting.")
                if _wandb is not None:
                    _wandb.finish()
                sys.exit(0)

        if args.train_only:
            print("[train-only] Phase-1 done (skipping collect/probe/vis).")
            if _wandb is not None:
                _wandb.finish()
            sys.exit(0)

        print(f"Saving agent → {agent_path}")
        leaves, treedef = jax.tree.flatten(state)
        #with open(agent_path, "wb") as f:
        #    pickle.dump(
        #        {"leaves": [np.array(l) for l in leaves], "treedef": treedef}, f
        #    )
    else:
        print(f"Loading agent ← {agent_path}")
        with open(agent_path, "rb") as f:
            saved = pickle.load(f)
        _, treedef = jax.tree.flatten(state)
        state = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

    if _memoryless:
        print("[memoryless diagnostic] done (skipping collect/probe/vis phases).")
        if _wandb is not None:
            _wandb.finish()
        sys.exit(0)

    # ── Phase 2 — Collect rollouts ───────────────────────────────────────────

    if not args.skip_collect:
        print(f"Board {args.rows}×{args.cols}, {N} cells.")

        def _save_dataset(path, c, b, hm, eb):
            print(f"  {len(c)} steps, {len(eb)-1} episodes → {path}")
            #with open(path, "wb") as f:
            #    pickle.dump({"carries": c, "board_masks": b, "hits_misses": hm,
            #                 "ep_bounds": eb, "rows": args.rows, "cols": args.cols,
            #                 "tag": tag}, f)

        print(f"Collecting {args.collect_steps} train steps (auto-reset)…")
        carries, board_masks, hits_misses, ep_bounds = _collect_and_parse(state, 1000, args.collect_steps)
        _save_dataset(dataset_path, carries, board_masks, hits_misses, ep_bounds)

        print(f"Collecting {args.collect_steps} test steps (separate seed)…")
        test_carries, test_board_masks, test_hits_misses, test_ep_bounds = _collect_and_parse(state, 2000, args.collect_steps)
        _save_dataset(test_dataset_path, test_carries, test_board_masks, test_hits_misses, test_ep_bounds)
    else:
        def _load_dataset(path):
            print(f"Loading dataset ← {path}")
            with open(path, "rb") as f:
                ds = pickle.load(f)
            return (ds["carries"], ds["board_masks"], ds["hits_misses"],
                    ds["ep_bounds"], ds["rows"], ds["cols"], ds.get("tag", tag))

        carries, board_masks, hits_misses, ep_bounds, _rows, _cols, tag = _load_dataset(dataset_path)
        test_carries, test_board_masks, test_hits_misses, test_ep_bounds, _, _, _ = _load_dataset(test_dataset_path)
        print(f"Train: {len(carries)} steps, {len(ep_bounds)-1} eps.  "
              f"Test: {len(test_carries)} steps, {len(test_ep_bounds)-1} eps.")

    # ── Phase 3 — Train probe ────────────────────────────────────────────────

    if not args.skip_probe:
        print(
            f"Training probe ({args.probe_steps} steps, "
            f"hidden={args.probe_hidden_dim}, lr={args.probe_lr})…"
        )
        probe_params = _train_probe(
            jnp.array(carries), _targets_from(board_masks, hits_misses),
            jax.random.key(args.seed + 2000),
            jax.random.key(args.seed + 3000),
            args.probe_steps, verbose=True,
        )

        print(f"Saving probe → {probe_path}")
        leaves, td = jax.tree.flatten(probe_params)
        #with open(probe_path, "wb") as f:
        #    pickle.dump({"leaves": [np.array(l) for l in leaves], "treedef": td}, f)
    else:
        print(f"Loading probe ← {probe_path}")
        with open(probe_path, "rb") as f:
            saved = pickle.load(f)
        ref = init_probe_params(
            jax.random.key(0), CARRY_DIM, 2 * N, args.probe_hidden_dim
        )
        _, td = jax.tree.flatten(ref)
        probe_params = td.unflatten([jnp.array(l) for l in saved["leaves"]])


# ════════════════════════════════════════════════════════════════════════════
#  Vis-only: load dataset + probe (no lambda-imitation needed)
# ════════════════════════════════════════════════════════════════════════════

if args.vis_only:
    print(f"Vis-only mode — loading artefacts from {args.output_dir}")

    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)
    carries = ds["carries"]
    board_masks = ds["board_masks"]
    ROWS, COLS = ds["rows"], ds["cols"]
    tag = ds.get("tag", "SAC+LD")
    N = ROWS * COLS
    CARRY_DIM = carries.shape[1]
    print(f"  train: {len(carries)} timesteps, {ROWS}x{COLS} board, "
          f"{len(ds['ep_bounds'])-1} episodes")

    with open(test_dataset_path, "rb") as f:
        tds = pickle.load(f)
    test_carries = tds["carries"]
    test_board_masks = tds["board_masks"]
    test_hits_misses = tds["hits_misses"]
    test_ep_bounds = tds["ep_bounds"]
    print(f"  test:  {len(test_carries)} timesteps, {len(test_ep_bounds)-1} episodes")

    with open(probe_path, "rb") as f:
        saved = pickle.load(f)
    probe_params = saved["treedef"].unflatten(
        [jnp.array(l) for l in saved["leaves"]]
    )
    print("  probe loaded")


# ════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Visualise (shared rendering code)
# ════════════════════════════════════════════════════════════════════════════

# Rendering helpers (matplotlib import, colours, render/to_grid and the
# _fig_* figure builders) live at module top so the periodic probe-eval can
# reuse them; the final visualisation below just calls the same functions.

# ── episode PNGs (from held-out test episodes) ─────────────────────────────

n_test_eps = len(test_ep_bounds) - 1
n_vis = min(args.vis_episodes, n_test_eps)
print(f"Generating {n_vis} episode visualisation(s) from test set…")
for vi in range(n_vis):
    p = _fig_episode(probe_params, test_carries, test_board_masks, test_hits_misses,
                     test_ep_bounds, vi,
                     os.path.join(args.output_dir, f"board_probe_ep{vi + 1}.png"),
                     tag, args.vis_frames)
    print(f"  → {p}")
    if _wandb is not None:
        _wandb.log({f"viz/episode_{vi + 1}": _wandb.Image(p)})

# ── accuracy summary ─────────────────────────────────────────────────────────
#
# A FIRED cell was directly observed by the memory (it chose that shot and saw
# the hit/miss), so decoding its ship/water label tests long-term RETENTION.
# An UNFIRED cell was never observed, so decoding it tests INFERENCE of the
# hidden board.  We report the two separately and, for fired cells, accuracy as
# a function of how long ago the cell was fired (the retention horizon).

print("Computing probe accuracy on held-out test set…")
test_logits = probe_forward(probe_params, jnp.array(test_carries))
test_probs_all = np.array(jax.nn.sigmoid(test_logits))       # (n, 2N)
test_probs = test_probs_all[:, :N]                           # (n, N) P(ship)
test_fired_probs = test_probs_all[:, N:]                     # (n, N) P(fired)
test_preds = (test_probs > 0.5).astype(np.float32)
test_targets = np.asarray(test_board_masks).astype(np.float32)
test_hm = np.asarray(test_hits_misses)
test_fired_targets = (test_hm != 0).astype(np.float32)       # (n, N) 1 = was fired at

correct = (test_preds == test_targets).astype(np.float32)   # (n, N)
fired = test_hm != 0                                        # observed cells
unfired = ~fired                                            # hidden cells


# (``_auroc`` is defined at module level, near ``probe_forward``.)


def _acc(mask):
    m = mask.astype(np.float32)
    return float((correct * m).sum() / max(m.sum(), 1.0))


acc = float(correct.mean())
ship_frac = float((test_targets == 1).mean())
majority_baseline = 1.0 - ship_frac  # accuracy of "predict water everywhere"


def _balanced(mask):
    """Balanced accuracy = mean(ship-recall, water-recall) within `mask`.

    Raw accuracy is misleading here: ships are only ~14% of a 10x10 board, so
    'predict water everywhere' already scores ~86%.  Balanced accuracy (and the
    per-class recalls) are the honest signal.
    """
    s = _acc(mask & (test_targets == 1))
    w = _acc(mask & (test_targets == 0))
    return 0.5 * (s + w), s, w


fired_acc = _acc(fired)
unfired_acc = _acc(unfired)
fired_bal, fired_ship, fired_water = _balanced(fired)
unfired_bal, unfired_ship, unfired_water = _balanced(unfired)
frac_fired = float(fired.mean())

# Threshold-free decodability (imbalance-proof): can the probe RANK ships above
# water?  0.5 = no signal regardless of where the 0.5 threshold lands.
fired_auroc = _auroc(test_probs[fired], test_targets[fired])
unfired_auroc = _auroc(test_probs[unfired], test_targets[unfired])

print(f"Ships are {ship_frac:.1%} of the board → majority (all-water) "
      f"baseline = {majority_baseline:.1%} accuracy.")
print(f"Overall: raw={acc:.1%}")
print(f"  Fired   (retention): AUROC={fired_auroc:.3f}  balanced={fired_bal:.1%}  "
      f"raw={fired_acc:.1%}  (hit→ship={fired_ship:.1%}, miss→water={fired_water:.1%})")
print(f"  Unfired (inference): AUROC={unfired_auroc:.3f}  balanced={unfired_bal:.1%}  "
      f"raw={unfired_acc:.1%}  (ship={unfired_ship:.1%}, water={unfired_water:.1%})")
print(f"  (AUROC 0.5 = no decodable signal; raw≈{majority_baseline:.0%} is just the prior.)")
print(f"  Avg {frac_fired:.0%} of cells fired per step.")

# Fired head: can the probe reconstruct *which* cells were fired at?  These were
# directly observed (the agent chose them), so high accuracy here is expected and
# acts as a sanity check that the memory retains its own action history.
test_fired_preds = (test_fired_probs > 0.5).astype(np.float32)
fired_pred_acc = float((test_fired_preds == test_fired_targets).mean())
fired_pred_auroc = _auroc(test_fired_probs, test_fired_targets)
fp_recall = float((test_fired_preds * test_fired_targets).sum()
                  / max(test_fired_targets.sum(), 1.0))      # P(pred fired | fired)
fp_specificity = float(((1 - test_fired_preds) * (1 - test_fired_targets)).sum()
                       / max((1 - test_fired_targets).sum(), 1.0))  # P(pred unfired | unfired)
print(f"Fired prediction (which cells were shot at): AUROC={fired_pred_auroc:.3f}  "
      f"acc={fired_pred_acc:.1%}  (recall={fp_recall:.1%}, specificity={fp_specificity:.1%})")

if _wandb is not None:
    _eval_metrics = {
        "eval/overall_acc": acc,
        "eval/ship_frac": ship_frac,
        "eval/majority_baseline": majority_baseline,
        "eval/fired_auroc": fired_auroc,
        "eval/fired_balanced": fired_bal,
        "eval/fired_acc": fired_acc,
        "eval/fired_ship_recall": fired_ship,
        "eval/fired_water_recall": fired_water,
        "eval/unfired_auroc": unfired_auroc,
        "eval/unfired_balanced": unfired_bal,
        "eval/unfired_acc": unfired_acc,
        "eval/unfired_ship_recall": unfired_ship,
        "eval/unfired_water_recall": unfired_water,
        "eval/frac_fired": frac_fired,
        "eval/fired_pred_auroc": fired_pred_auroc,
        "eval/fired_pred_acc": fired_pred_acc,
        "eval/fired_pred_recall": fp_recall,
        "eval/fired_pred_specificity": fp_specificity,
    }
    _wandb.log(_eval_metrics)
    _wandb.summary.update(_eval_metrics)


p = _fig_accuracy(probe_params, test_carries, test_board_masks, test_hits_misses,
                  os.path.join(args.output_dir, "board_probe_accuracy.png"), tag)
print(f"  → {p}")
if _wandb is not None:
    _wandb.log({"viz/per_cell_accuracy": _wandb.Image(p)})

# ── retention horizon: fired-cell accuracy vs steps-since-fired ──────────────
#
# For each fired cell at step t, its "recency" is t − (step it was first fired).
# recency 0 = just fired (trivially observed); larger recency tests how long the
# memory holds the observation.  A smaller memory should decay sooner.

p = _fig_retention(probe_params, test_carries, test_board_masks, test_hits_misses,
                   test_ep_bounds, os.path.join(args.output_dir, "board_probe_retention.png"),
                   tag)
if p is not None:
    print(f"  → {p}")
    if _wandb is not None:
        _wandb.log({"viz/retention_horizon": _wandb.Image(p)})
else:
    print("  (no fired cells collected — skipping retention plot)")

# ── mp4 / gif ────────────────────────────────────────────────────────────────

if args.mp4:
    if not args.vis_only:
        # Roll out the *deterministic* (greedy, masked) policy for one fresh
        # episode — not the epsilon-soft policy used for data collection.
        print("Rolling out the deterministic policy for one episode (mp4)…")

        @jax.jit
        def _det_rollout(agent_state, key):
            key, rk = jax.random.split(key)
            obs, est = env.reset(rk, env_params)
            carry = zero_carry()
            prev_action = zero_prev_action()

            def step_fn(s, _):
                obs, est, carry, prev_action, key, alive = s
                key, sk, ek = jax.random.split(key, 3)
                raw, nc = fns.predict(
                    agent_state, obs, carry, sk, deterministic=True,
                    prev_action=prev_action,
                )
                a = jnp.round(raw).astype(jnp.int32)
                _nobs, nst, _r, d, _ = env.step(ek, est, a, env_params)
                npa = jax.nn.one_hot(a, N)
                valid = alive  # 1 up to and including the terminal step, 0 after
                new_alive = alive * (1.0 - d.astype(jnp.float32))
                npa = jnp.where(d, zero_prev_action(), npa)
                frame = {
                    # ``nc`` has processed ``obs`` (the result of a_{t-1}), so the
                    # memory knows the board state through a_{t-1} — NOT a_t, whose
                    # result is only observed next step.  Pair it with the *pre*-step
                    # state ``est`` (shots through a_{t-1}), exactly as the collection
                    # phase does (env_st read before env.step); using the post-step
                    # ``nst`` would put the fired-cell truth one shot ahead of the
                    # carry (the off-by-one in the retention overlay).  Board (ships)
                    # is static within an episode, so est/nst agree there.
                    "carries": nc[:CARRY_DIM],
                    "board": est.board.reshape(-1).astype(jnp.float32),
                    "hits": est.hits_misses.reshape(-1).astype(jnp.float32),
                    "valid": valid,
                }
                return (_nobs, nst, nc, npa, key, new_alive), frame

            init = (obs, est, carry, prev_action, key, jnp.float32(1.0))
            _, data = jax.lax.scan(step_fn, init, length=_MAX_STEPS)
            return data

        data = _det_rollout(state, jax.random.key(args.seed + 7))
        valid = np.array(data["valid"])
        ep_len = int(valid.sum())
        ep_c = np.array(data["carries"])[:ep_len]
        ep_b = np.array(data["board"])[:ep_len]
        ep_hm = np.array(data["hits"])[:ep_len]
        movie_tag = "deterministic policy"
    else:
        # vis-only has no agent/env loaded — fall back to the longest collected
        # (epsilon-soft) test episode.
        print("vis-only: no agent loaded — using the longest collected test episode.")
        test_ep_lens = np.diff(test_ep_bounds)
        best = int(np.argmax(test_ep_lens))
        start, end = int(test_ep_bounds[best]), int(test_ep_bounds[best + 1])
        ep_c = test_carries[start:end]
        ep_b = test_board_masks[start:end]
        ep_hm = test_hits_misses[start:end]
        ep_len = end - start
        movie_tag = "epsilon-soft (collected)"

    print(f"Rendering mp4 ({ep_len} frames, {movie_tag})…")
    movie_path = _fig_movie(probe_params, ep_c, ep_b, ep_hm,
                            os.path.join(args.output_dir, "board_probe"), tag, movie_tag)
    print(f"  → {movie_path}")
    if _wandb is not None and movie_path is not None:
        _wandb.log({"viz/episode_movie": _wandb.Video(movie_path)})

if _wandb is not None:
    _wandb.finish()

print("Done.")
