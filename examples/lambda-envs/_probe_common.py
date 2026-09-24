"""Env-agnostic machinery shared by the probe scripts.

``battleship_board_probe.py`` and ``pocman_pellet_probe.py`` differ in the env,
in which hidden state is probed and in how it is drawn.  Everything else — the
CLI, the W&B wiring, the probe MLP, the saturation-resistant metrics, the
multi-seed plumbing, the FE checkpointing and the two evaluators — is the same
code, and lives here.

Nothing in this module imports an environment or a renderer: each probe script
supplies those, plus the two env-specific callables the probe needs per step,

    truth      what the probe must decode (the hidden state), and
    observed   which of those cells the memory has actually seen,

which is what splits the metrics into retention (observed) and inference
(never observed).
"""

import os
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────

def add_common_args(parser, *, output_dir_default, wandb_project_default,
                    include_probe=True):
    """Add every flag that is not environment-specific.

    The caller adds its own environment group (board size, ship lengths, maze
    options, …) and its network-architecture flags before calling this.

    ``include_probe=False`` omits the collection, probe and visualisation flags,
    for environments where training and evaluation are the whole story and there
    is no hidden state worth decoding.  Everything else is unchanged.
    """
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
    g.add_argument("--final-return-window", type=int, default=3,
                   help="number of final rounds averaged for the smoothed final "
                        "return / steps-to-clear (default 10).")
    g.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
    g.add_argument("--memory-hidden-dim", type=int, default=512)
    g.add_argument("--projection-dim", type=int, default=128)
    g.add_argument("--fe-lr", type=float, default=1e-4, help="feature-extractor lr (default 1e-4)")
    g.add_argument("--actor-lr", type=float, default=1e-4, help="actor lr (default 1e-4)")
    g.add_argument("--critic-lr", type=float, default=1e-4, help="critic lr (default 2e-4)")
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
    g.add_argument("--critic-layer-norm", dest="critic_layer_norm", action="store_true",
                   help="use LayerNorm in the critic MLP (default on).")
    g.add_argument("--no-critic-layer-norm", dest="critic_layer_norm", action="store_false",
                   help="disable critic LayerNorm (control: LN can wash out small "
                        "per-action Q differences).")
    parser.set_defaults(critic_layer_norm=True)
    g.add_argument("--behaviour-epsilon", type=float, default=0,
                   help="fraction of online-collection steps driven by a scripted "
                        "behaviour policy instead of the actor (0 = off, default).")
    g.add_argument("--approximate-lambda", dest="approximate_lambda", action="store_true")
    g.add_argument("--no-approximate-lambda", dest="approximate_lambda", action="store_false")
    parser.set_defaults(approximate_lambda=True)
    g.add_argument("--gvd", dest="gvd", action="store_true",
                   help="enable the GVD successor-feature branches (reward-free "
                        "memory pressure; an agent trained with --gvd must be "
                        "reloaded with --gvd)")
    g.add_argument("--no-gvd", dest="gvd", action="store_false")
    parser.set_defaults(gvd=False)
    g.add_argument("--gvd-coef", type=float, default=0.2, help="GVD discrepancy coefficient (default 1.0)")
    g.add_argument("--gvd-features", type=int, default=16,
                   help="random-projection width of the GVD feature map; total dim "
                        "is 1 (hit bit) + N (default 16)")
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
                   help="stop-gradient the FE latents feeding the SAC twin-Q "
                        "critic: its Bellman regression no longer backprops into "
                        "the shared memory.  The λ-critics and the λ-discrepancy "
                        "keep training the FE, so this isolates SAC "
                        "value-magnitude inflation as a memory-wrecking path.")
    parser.set_defaults(stop_critic_fe=False)
    g.add_argument("--ld-center", dest="ld_center", action="store_true",
                   help="minimise var(Q1-Q2) instead of mean((Q1-Q2)^2): subtract "
                        "the batch mean from the λ-discrepancy first.  The "
                        "uncentred term is dominated by a constant offset between "
                        "the λ-critics (state-independent, so it teaches the memory "
                        "nothing) that grows with their value drift.")
    parser.set_defaults(ld_center=False)
    g.add_argument("--retrace", dest="retrace", action="store_true",
                   help="regress the λ-critics onto Retrace(λ) targets instead of "
                        "V-trace ones.  V-trace scales the current step's delta by "
                        "ρ, so with a near-deterministic actor (ρ=0 on ~97%% of "
                        "transitions here) the target loses the reward entirely and "
                        "the value level drifts; Retrace keeps the 1-step term at "
                        "coefficient 1 and only cuts multi-step propagation.")
    parser.set_defaults(retrace=True)
    g.add_argument("--offline", action="store_true",
                   help="train fully offline: fill the buffer once (see "
                        "--expert-prefill-steps) and then run gradient updates with "
                        "NO environment interaction.  --rounds x --train-steps then "
                        "counts updates rather than env steps.  Everything else — "
                        "multi-seed vmapping, probe-evals, W&B aggregation, "
                        "checkpointing — is unchanged.")
    g.add_argument("--probe-rollout-policy", choices=("actor", "expert", "bayes"),
                   default="actor",
                   help="which policy generates the probe's rollouts (default "
                        "actor).  'expert' probes along the scripted player's "
                        "trajectories, i.e. the distribution an offline agent was "
                        "trained on, which the actor's own rollouts do not match "
                        "('bayes' is the old spelling of 'expert').")
    g.add_argument("--save-fe-every-eval", action="store_true",
                   help="save each seed's feature extractor at every probe-eval as "
                        "fe_seed<i>_<step>.pkl, for transplanting with --init-fe")
    g.add_argument("--init-fe", default=None, metavar="PATH",
                   help="initialise the feature extractor (and its EMA target) from "
                        "a saved fe_*.pkl.  Pair with --fe-lr 0 to freeze that "
                        "memory and train only the heads on top of it.")
    g.add_argument("--expert-prefill-steps", type=int, default=0,
                   help="fill the replay buffer with this many transitions from "
                        "the scripted expert player before online training "
                        "starts, instead of the uniform-random prefill (0 = random, "
                        "the default).  Tests whether an expert warm start gets the "
                        "agent past the chicken-and-egg where memory only pays off "
                        "once the policy plays well.")
    g.add_argument("--expert-prefill-epsilon", type=float, default=0.1,
                   help="epsilon-greedy rate of that expert prefill (default 0.1)")
    g.add_argument("--critic-greedy-eval", dest="critic_greedy_eval",
                   action="store_true",
                   help="each round, also evaluate the CRITIC-greedy policy "
                        "(argmax_a Q over legal actions) and log its return, "
                        "steps-to-clear, and the per-action spread qstd / qrange.  "
                        "Splits a memory failure from an extraction failure "
                        "(default on; read-only, no effect on training).")
    g.add_argument("--no-critic-greedy-eval", dest="critic_greedy_eval",
                   action="store_false")
    parser.set_defaults(critic_greedy_eval=False)
    g.add_argument("--actor-critic", choices=("sac", "lambda1", "lambda2"),
                   default="sac",
                   help="which critic the actor is extracted from (default sac).  "
                        "The λ-critics are the calibrated ones on this data once "
                        "Retrace + the twin fix + a small --lambda-coef are in "
                        "(+47 against a +35.6 fixed point), while the SAC critic "
                        "overestimates past the +100 return ceiling; picking a "
                        "λ-critic also couples the policy to the λ-discrepancy.")
    g.add_argument("--lambda-coef", type=float, default=1.0,
                   help="weight on the λ-discrepancy term (default 1.0, which is "
                        "what every run so far used).  Note the reference "
                        "implementation uses a convex combination instead — "
                        "ld_weight·LD + (1-ld_weight)·value_loss with ld_weight in "
                        "{0.125, 0.25, 0.5} and 0.5 selected for Battleship-10 — so "
                        "an additive 1.0 on top of the value losses is a different "
                        "and much heavier weighting, especially once ld inflates.")
    g.add_argument("--per-alpha", type=float, default=0.0,
                   help="prioritised sequence replay: draw windows with "
                        "probability ∝ priority^alpha, where priority is the "
                        "window's trace mass (geometric mean of its clipped "
                        "importance ratios).  0 = uniform (default).  Targets the "
                        "λ-discrepancy: with ρ=0 on most transitions every trace "
                        "is cut and λ1/λ2 collapse to the same target, so only the "
                        "windows where the policies agree carry any signal.")
    g.add_argument("--per-beta", type=float, default=0.4,
                   help="importance-sampling exponent correcting the prioritised "
                        "draw (weights normalised by their batch max; default 0.4)")
    g.add_argument("--per-window", type=int, default=0,
                   help="window PREFIX length the priority is computed over "
                        "(0 = the whole --lambda-truncation span).  Measured: over "
                        "50 steps the grounded fraction concentrates and the score "
                        "is flat across windows (ESS 1.00).  The trace is "
                        "multiplicative, so survival of the FIRST few steps is what "
                        "varies — 4 to 8 is the useful range.")
    g.add_argument("--per-ratio-floor", type=float, default=1e-3,
                   help="lower clip on each ratio in the priority (default 1e-3)")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--sequence-length", type=int, default=50)
    g.add_argument("--burn-in-length", type=int, default=10)
    g.add_argument("--burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
                   action="store_true",
                   help="store the online carry per transition and initialise the "
                        "training burn-in from it instead of zeros (R2D2 "
                        "stored-state; enables shorter --burn-in-length; costs "
                        "carry_dim x buffer_size x 4 bytes extra)")
    g.add_argument("--no-burn-in-from-stored-carry", dest="burn_in_from_stored_carry",
                   action="store_false")
    parser.set_defaults(burn_in_from_stored_carry=False)
    g.add_argument("--online-buffer-size", type=int, default=100_000)
    g.add_argument("--use-sac", dest="use_sac", action="store_true",
                   help="whether or not to use a SAC entropy term for update"
                        "or just use entropy globally in loss")
    g.add_argument("--no-use-sac", dest="use_sac", action="store_false")
    parser.set_defaults(use_sac=False)

    g = parser.add_argument_group("I/O")
    g.add_argument("--output-dir",
                   default=os.environ.get("PROBE_OUTPUT_DIR", output_dir_default),
                   help="where artefacts go (default $PROBE_OUTPUT_DIR, else "
                        f"{output_dir_default!r}).  The env var exists for W&B\n"
                        "sweeps: the agent runs the command from the sweep config, "
                        "so the only way to redirect output per machine is the "
                        "environment.  Each sweep run still gets its own "
                        "run_<id> subdirectory underneath.")
    g.add_argument("--skip-train", action="store_true", help="load agent from output-dir")
    g.add_argument("--train-only", action="store_true",
                   help="stop after training (skip any collect/probe/visualise)")
    g.add_argument("--save-checkpoint-at", type=int, default=0, metavar="ROUND",
                   help="save a resumable checkpoint after this round (0 = off)")
    g.add_argument("--resume-from", default=None, metavar="PATH",
                   help="resume training from a checkpoint written by "
                        "--save-checkpoint-at")

    if include_probe:
        g = parser.add_argument_group("data collection")
        g.add_argument("--collect-steps", type=int, default=100_000, help="total env steps to collect (default 100 000)")
        g.add_argument("--collect-epsilon", type=float, default=0.1, help="epsilon-greedy rate during collection (default 0.1)")

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
        parser.set_defaults(probe_eval_vis=False)

        g = parser.add_argument_group("probe I/O & visualisation")
        g.add_argument("--skip-collect", action="store_true", help="load dataset from output-dir")
        g.add_argument("--skip-probe", action="store_true", help="load probe from output-dir")
        g.add_argument("--setup-only", action="store_true",
                       help="build the env / agent / probe helpers, then exit before "
                            "the training phase, so an importing script can drive its "
                            "own pipeline from this module's globals")
        g.add_argument("--vis-only", action="store_true",
                       help="skip phases 1-3 and render from saved artefacts "
                            "(needs only jax + matplotlib)")
        g.add_argument("--mp4", action="store_true", help="also write an animation")
        g.add_argument("--vis-episodes", type=int, default=3, help="episodes to render (default 3)")
        g.add_argument("--vis-frames", type=int, default=8, help="frames per episode figure (default 8)")

    g = parser.add_argument_group("logging")
    g.add_argument("--wandb", action="store_true", help="log metrics/images to Weights & Biases")
    g.add_argument("--wandb-project", default=wandb_project_default)
    g.add_argument("--wandb-run-name", default=None)
    return parser

def parse_args(parser, *, extra_flags_env):
    """Parse argv strictly: rewrite underscore flags, reject unknown ones.

    W&B's backend rejects sweep parameter names containing hyphens ("sweep
    config: ignoring unknown parameter 'batch-size'") and silently drops them, so
    a sweep can only name parameters with underscores -- which ${args} then emits
    as --batch_size=... .  Accept that spelling by rewriting it to the canonical
    hyphenated flag, so sweeps and humans can both be right.

    Unknown arguments are rejected rather than ignored.  parse_known_args is
    required because another script may share this argv (an importing script
    drives this module with its own flags present), but silently dropping a
    misspelled flag is worse: a W&B sweep passing --probe_eval_interval=0 (the
    parameter name, underscores) instead of --probe-eval-interval=0 would run the
    whole sweep at the default and look perfectly healthy.  An importing script
    declares its own flags in the ``extra_flags_env`` environment variable.
    """
    _known_opts = {opt for action in parser._actions for opt in action.option_strings}
    sys.argv[1:] = [
        (lambda head, sep, tail: (head.replace("_", "-") + sep + tail
                                  if head.startswith("--")
                                  and head.replace("_", "-") in _known_opts
                                  else tok))(*tok.partition("="))
        for tok in sys.argv[1:]
    ]

    args, _unknown = parser.parse_known_args()

    _allowed_extra = {f for f in os.environ.get(extra_flags_env, "").split(",") if f}

    def _is_flag(tok):
        if not tok.startswith("-"):
            return False
        try:                      # negative numbers are values, not flags
            float(tok)
            return False
        except ValueError:
            return True

    _bad = [t for t in _unknown if _is_flag(t) and t.split("=")[0] not in _allowed_extra]
    if _bad:
        # Quote them: a token containing a space is almost always a W&B sweep
        # `command:` entry written as "- --flag value", which arrives as ONE argv
        # element.  Unquoted, the message reads like a perfectly normal command
        # line and the real problem is invisible.
        hint = ""
        if any(" " in t for t in _bad):
            hint = ("\nOne of these contains a space, so it arrived as a single "
                    "argument.  In a W&B sweep `command:` list, write "
                    "`- --flag=value` (or put the flag and its value on separate "
                    "lines) — each list entry becomes one argv element.")
        sys.exit(
            f"{parser.prog}: unrecognised argument(s): "
            f"{' '.join(repr(t) for t in _bad)}\n"
            "Note flags use hyphens, not underscores (e.g. --probe-eval-interval). "
            f"If another script owns these flags, list them in {extra_flags_env}."
            + hint
        )
    return args

def apply_sweep_config(parser, args):
    """Under ``wandb agent``, override args from the sweep config.

    The hyperparameters arrive via the sweep config, not the CLI: the agent
    passes underscore-style ``--name=value`` args that match neither the dashed
    flags nor the store_true/store_false pairs (parse_known_args drops them).
    Attach to the run the agent pre-created and override args from its config
    instead.  Returns the sweep run (or None).
    """
    if not os.environ.get("WANDB_SWEEP_ID"):
        return None
    import wandb as _wandb_mod

    sweep_run = _wandb_mod.init()
    _SWEEP_ALIASES = {"use_gvd": "gvd"}
    _SWEEP_IGNORED = {"action_masked", "buffer_size"}  # not script knobs
    _ARG_TYPES = {a.dest: a.type for a in parser._actions if callable(a.type)}
    for _k, _v in dict(sweep_run.config).items():
        _key = _SWEEP_ALIASES.get(_k, _k)
        if _key in _SWEEP_IGNORED:
            continue
        if not hasattr(args, _key):
            print(f"sweep config: ignoring unknown parameter {_k!r}")
            continue
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
    args.output_dir = os.path.join(args.output_dir, f"run_{sweep_run.id}")
    print(f"sweep run {sweep_run.id}: output_dir → {args.output_dir}")
    return sweep_run

def common_wandb_config(args):
    """The env-independent half of the W&B config; the caller adds its own."""
    return {
        "fe_lr": args.fe_lr,
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
        "approximate_lambda": args.approximate_lambda,
        "use_gvd": args.gvd,
        "gvd_coef": args.gvd_coef,
        "gvd_features": args.gvd_features,
        "gvd_lambda1": args.gvd_lambda1,
        "gvd_lambda2": args.gvd_lambda2,
        "gvd_sf_lr": args.gvd_sf_lr,
        "gvd_stop_fe": args.gvd_stop_fe,
        "stop_actor_fe": args.stop_actor_fe,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "burn_in_length": args.burn_in_length,
        "burn_in_from_stored_carry": args.burn_in_from_stored_carry,
        "online_buffer_size": args.online_buffer_size,
        "rounds": args.rounds,
        "train_steps": args.train_steps,
        "collect_steps": getattr(args, "collect_steps", None),
        "collect_epsilon": getattr(args, "collect_epsilon", None),
        "probe_steps": getattr(args, "probe_steps", None),
        "probe_lr": getattr(args, "probe_lr", None),
        "probe_hidden_dim": getattr(args, "probe_hidden_dim", None),
        "probe_batch_size": getattr(args, "probe_batch_size", None),
        "probe_eval_interval": getattr(args, "probe_eval_interval", None),
        "probe_eval_steps": getattr(args, "probe_eval_steps", None),
        "probe_eval_collect_steps": getattr(args, "probe_eval_collect_steps", None),
        "use_sac": args.use_sac,
        "stop_critic_fe": args.stop_critic_fe,
        "ld_center": args.ld_center,
        "retrace": args.retrace,
        "lambda_coef": args.lambda_coef,
        "actor_critic": args.actor_critic,
        "per_alpha": args.per_alpha,
        "per_beta": args.per_beta,
        "num_seeds": args.num_seeds,
        "concurrent_seeds": args.concurrent_seeds,
        "final_return_window": args.final_return_window,
    }

def init_wandb(args, config, sweep_run):
    """Start (or attach to) the W&B run and declare the metric axes.

    Agent training is keyed to ``env_interactions`` and probe training to
    ``probe_step`` via define_metric, so the two phases get separate x-axes and
    we never pass an explicit step= (which would clash across the two phases).
    Vis-only mode is portable (no agent/env), so wandb stays off there.
    Returns the wandb module, or None when logging is off.
    """
    if not args.wandb or getattr(args, "vis_only", False):
        return None
    try:
        import wandb as _wandb
    except ImportError:
        sys.exit("wandb not installed.  Install with:  pip install wandb")

    if sweep_run is not None:
        # The sweep agent already created (and owns) the run — reuse it; just
        # merge our config so the dashboard shows the resolved hyperparameters.
        sweep_run.config.update(config, allow_val_change=True)
    else:
        _wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
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
    return _wandb

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

def auroc(scores, labels):
    """Threshold-free separability: P(score[positive] > score[negative]).

    Rank-based (Mann-Whitney U); robust to the class imbalance that makes raw
    accuracy and recall@0.5 misleading.  0.5 = no signal, 1.0 = perfectly ranks
    positives above negatives.  NaN if a class is absent.
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

def make_probe_trainer(optimiser, *, carry_dim, n_out, hidden, batch_size,
                       tqdm=None, wandb=None):
    """Build the (jitted) probe SGD chunk plus its single- and multi-seed drivers.

    One optimiser / jitted chunk, reused across every probe trained (the final
    one and each periodic eval), so the chunk compiles only once.
    """
    import optax  # lazy: --vis-only renders with just jax + matplotlib

    @partial(jax.jit, static_argnames=["n_steps", "batch_size"])
    def train_chunk(params, opt_state, key, c_data, t_data, n_steps, batch_size):
        n = c_data.shape[0]

        def body(carry, _):
            params, opt_state, key = carry
            key, bk = jax.random.split(key)
            idx = jax.random.randint(bk, (batch_size,), 0, n)

            def loss_fn(p):
                logits = probe_forward(p, c_data[idx])
                return optax.sigmoid_binary_cross_entropy(logits, t_data[idx]).mean()

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_os = optimiser.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), new_os, key), loss

        (params, opt_state, key), losses = jax.lax.scan(
            body, (params, opt_state, key), length=n_steps)
        return params, opt_state, key, losses[-1]

    _CHUNK = 50_000

    def train(c_jnp, t_jnp, init_key, train_key, n_steps, verbose=False):
        params = init_probe_params(init_key, carry_dim, n_out, hidden)
        opt_state = optimiser.init(params)
        n_chunks = n_steps // _CHUNK
        remainder = n_steps % _CHUNK
        key = train_key
        steps_done = 0

        def _step(nsteps):
            nonlocal params, opt_state, key, steps_done
            params, opt_state, key, loss_val = train_chunk(
                params, opt_state, key, c_jnp, t_jnp, nsteps, batch_size)
            steps_done += nsteps
            if verbose:
                print(f"  step {steps_done:7d}  bce={float(loss_val):.6f}")
                if wandb is not None:
                    wandb.log({"probe_step": steps_done, "probe/bce": float(loss_val)})

        for _ in (tqdm(range(n_chunks), desc="Probe") if (verbose and tqdm) else range(n_chunks)):
            _step(_CHUNK)
        if remainder > 0:
            _step(remainder)
        return params

    def train_v(c_b, t_b, init_keys, train_keys, n_steps):
        """Vmapped probe training: (S, n, carry_dim)/(S, n, n_out) → batched params."""
        params = jax.vmap(
            lambda k: init_probe_params(k, carry_dim, n_out, hidden))(init_keys)
        opt_state = jax.vmap(optimiser.init)(params)
        n_chunks = n_steps // _CHUNK
        remainder = n_steps % _CHUNK
        keys = train_keys

        def _chunk_v(p, o, k, nsteps):
            # vmap over params/opt/key AND the per-seed data slices (c_b/t_b
            # are (S, n, …) — each lane must train on its own slice).
            return jax.vmap(
                lambda p, o, k, c, t: train_chunk(p, o, k, c, t, nsteps, batch_size),
                in_axes=(0, 0, 0, 0, 0),
            )(p, o, k, c_b, t_b)

        for _ in range(n_chunks):
            params, opt_state, keys, _loss = _chunk_v(params, opt_state, keys, _CHUNK)
        if remainder > 0:
            params, opt_state, keys, _loss = _chunk_v(params, opt_state, keys, remainder)
        return params

    return train, train_v, train_chunk

# ── probe metrics ────────────────────────────────────────────────────────────
#
# An OBSERVED cell was directly seen by the memory (the agent acted on it and
# saw the outcome), so decoding its label tests long-term RETENTION.  An
# UNOBSERVED cell was never seen, so decoding it tests INFERENCE of the hidden
# state.  The two are reported separately and, for observed cells, as a function
# of how long ago they were observed (the retention horizon).

AGE_BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 35), (36, 10 ** 6)]

def cell_ages(observed, ep_bounds):
    """Steps since each observed cell was first observed (-1 if never).

    A cell's outcome enters the memory once, when it is observed; how long ago
    that was is what separates retained memory from a short echo.  An untrained
    GRU decodes age 0-2 cells near-perfectly and collapses to chance by age ~10,
    so an aggregate over all observed cells is dominated by recency rather than
    by memory.
    """
    observed = np.asarray(observed)
    ages = -np.ones_like(observed, dtype=np.float32)
    for s0, s1 in zip(ep_bounds[:-1], ep_bounds[1:]):
        seg = observed[s0:s1] != 0                # (T, N) observed-so-far mask
        if len(seg) == 0:
            continue
        first = np.argmax(seg, axis=0)            # first step each cell is observed
        ever = seg.any(axis=0)
        t = np.arange(len(seg))[:, None]
        a = t - first[None, :]
        ages[s0:s1] = np.where(seg & ever[None, :], a, -1)
    return ages

def probe_metrics_from_probs(probs_all, t_truth, t_observed, t_eb=None, *, n_cells):
    """Headline decodability metrics from precomputed host probabilities.

    ``probs_all`` is (T, 2·n_cells): the first half is P(hidden state), the
    second half P(this cell was observed).  Split from a probe-params entry
    point so the multi-seed path can score from predictions already pulled to
    host (pure numpy — no GPU op), which is what lets the matplotlib rendering
    overlap the next training round.
    """
    probs_all = np.asarray(probs_all)
    probs = probs_all[:, :n_cells]            # P(hidden state)
    fired_probs = probs_all[:, n_cells:]      # P(observed)
    preds = (probs > 0.5).astype(np.float32)
    targets = np.asarray(t_truth).astype(np.float32)
    hm = np.asarray(t_observed)
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

    # AUROC saturates well before the memory stops improving (0.96 -> 0.9998
    # spans the entire interesting range), so report quantities that stay
    # discriminative up there and separate recency from retention.
    n_wrong = ((1.0 - correct) * fired).sum(axis=-1)
    _eps = 1e-7
    p_truth = np.where(targets == 1, probs, 1.0 - probs)
    bits = -np.log2(np.clip(p_truth[fired], _eps, 1.0))
    prior = targets[fired].mean() if fired.any() else 0.5
    prior_bits = -(prior * np.log2(max(prior, _eps))
                   + (1 - prior) * np.log2(max(1 - prior, _eps)))
    extra = {
        "errors_per_state": float(n_wrong.mean()),
        "exact_match": float((n_wrong == 0).mean()),
        "bits_per_cell": float(bits.mean()) if fired.any() else float("nan"),
        "info_gain_bits": float(prior_bits - bits.mean()) if fired.any() else float("nan"),
    }
    if t_eb is not None:
        ages = cell_ages(t_observed, np.asarray(t_eb))
        horizon = 0.0
        for lo, hi in AGE_BUCKETS:
            m = fired & (ages >= lo) & (ages <= hi)
            b = _balanced(m) if m.sum() else float("nan")
            extra[f"bal_age{lo}_{hi if hi < 10 ** 6 else 'plus'}"] = b
            if m.sum() and b >= 0.9:
                horizon = min(hi, 50)
        # oldest age bucket still decoded at >=90% balanced recall
        extra["horizon_steps"] = horizon

    return {
        **extra,
        "overall_acc": float(correct.mean()),
        "fired_auroc": auroc(probs[fired], targets[fired]),
        "fired_balanced": _balanced(fired),
        "fired_acc": _acc(fired),
        "unfired_auroc": auroc(probs[unfired], targets[unfired]),
        "unfired_balanced": _balanced(unfired),
        "unfired_acc": _acc(unfired),
        "frac_fired": float(fired.mean()),
        "fired_pred_auroc": auroc(fired_probs, fired_targets),
        "fired_pred_acc": float((fired_pred == fired_targets).mean()),
    }

def probe_metrics(probe_params, t_carries, t_truth, t_observed, t_eb=None, *, n_cells):
    """Headline decodability metrics on a held-out test set (no plots)."""
    probs_all = np.array(jax.nn.sigmoid(probe_forward(probe_params, jnp.array(t_carries))))
    return probe_metrics_from_probs(probs_all, t_truth, t_observed, t_eb, n_cells=n_cells)

def probe_metrics_visitation(probs_all, t_truth, t_eb=None):
    """Metrics for a target the agent itself consumes (PocMan pellets).

    Battleship's split does not apply here: a pellet is gone precisely because
    the agent ate it, so ``truth`` and "has this cell been observed" are
    complements and a fired/unfired split would leave one class per group.  What
    the memory is actually being asked is "which cells have I already cleared",
    so the headline is decodability over ALL cells and the breakdown is by
    RECENCY — how long ago the cell was eaten — which is the retention horizon.

    ``probs_all`` is (T, n_cells) = P(pellet still there).
    """
    probs_all = np.asarray(probs_all)
    preds = (probs_all > 0.5).astype(np.float32)
    targets = np.asarray(t_truth).astype(np.float32)
    correct = (preds == targets).astype(np.float32)
    eaten = targets == 0          # cells the agent has visited
    alive = ~eaten

    def _recall(mask):
        m = mask.astype(np.float32)
        return float((correct * m).sum() / max(m.sum(), 1.0))

    n_wrong = (1.0 - correct).sum(axis=-1)
    _eps = 1e-7
    p_truth = np.where(targets == 1, probs_all, 1.0 - probs_all)
    bits = -np.log2(np.clip(p_truth, _eps, 1.0))
    prior = float(targets.mean())
    prior_bits = -(prior * np.log2(max(prior, _eps))
                   + (1 - prior) * np.log2(max(1 - prior, _eps)))
    out = {
        "errors_per_state": float(n_wrong.mean()),
        "exact_match": float((n_wrong == 0).mean()),
        "bits_per_cell": float(bits.mean()),
        "info_gain_bits": float(prior_bits - bits.mean()),
        "overall_acc": float(correct.mean()),
        "auroc": auroc(probs_all, targets),
        "balanced": 0.5 * (_recall(alive) + _recall(eaten)),
        "eaten_recall": _recall(eaten),
        "uneaten_recall": _recall(alive),
        "frac_eaten": float(eaten.mean()),
    }
    if t_eb is not None:
        # age = steps since the cell was eaten (eaten cells only)
        ages = cell_ages(eaten.astype(np.float32), np.asarray(t_eb))
        horizon = 0.0
        for lo, hi in AGE_BUCKETS:
            m = eaten & (ages >= lo) & (ages <= hi)
            # only one class is present per bucket, so this is recall, and the
            # comparable quantity to Battleship's balanced recall by age
            b = _recall(m) if m.sum() else float("nan")
            out[f"bal_age{lo}_{hi if hi < 10 ** 6 else 'plus'}"] = b
            if m.sum() and b >= 0.9:
                horizon = min(hi, 50)
        out["horizon_steps"] = horizon
    return out

def agg(values, prefix):
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

# ── multi-seed plumbing ──────────────────────────────────────────────────────
#
# Concurrent seeds are trained in one vmapped+jitted kernel: per-seed agent
# states are stacked along a leading axis, vmapped over, then split back out.

def stack_states(states):
    """Stack a list of per-seed pytrees along a new leading axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)

def unstack_state(batched, j):
    """Pull seed ``j``'s pytree out of a leading-axis-batched state."""
    return jax.tree.map(lambda x: x[j], batched)

def split_each(keys):
    """Split a batch of PRNG keys, returning two batches (carry, fresh)."""
    out = jax.vmap(lambda k: jax.random.split(k))(keys)
    return out[:, 0], out[:, 1]

def episode_bounds(dones, length):
    """Indices delimiting complete episodes in a flat rollout."""
    di = np.where(np.asarray(dones) > 0.5)[0]
    es = np.concatenate([[0], di + 1])
    es = es[es < length]
    return np.concatenate([es, [length]])

# ── feature-extractor checkpointing ──────────────────────────────────────────

def make_fe_checkpointer(init_fe_path, output_dir):
    """Build (transplant_fe, save_fe) for --init-fe / --save-fe-every-eval."""
    payload = None
    if init_fe_path:
        with open(init_fe_path, "rb") as f:
            payload = pickle.load(f)
        print(f"initialising every seed's FE from {init_fe_path} "
              f"(saved at {payload.get('updates', '?')} updates)")

    def transplant_fe(state):
        """Replace the FE and its EMA target with a saved memory.

        Only the feature extractor moves: the actor and critics start fresh, so
        what is tested is what the memory alone supports.
        """
        if payload is None:
            return state
        tree = jax.tree.structure(state.feature_extractor)
        return state._replace(
            feature_extractor=jax.tree.unflatten(
                tree, [jnp.asarray(x) for x in payload["fe"]]),
            feature_extractor_target=jax.tree.unflatten(
                tree, [jnp.asarray(x) for x in payload["fe_target"]]),
        )

    def save_fe(state, seed_idx, step):
        path = os.path.join(output_dir, f"fe_seed{seed_idx}_{step}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "fe": [np.asarray(x) for x in jax.tree.leaves(state.feature_extractor)],
                "fe_target": [np.asarray(x)
                              for x in jax.tree.leaves(state.feature_extractor_target)],
                "updates": step, "seed": seed_idx,
            }, f)
        return path

    return transplant_fe, save_fe

# ── evaluators ───────────────────────────────────────────────────────────────

def make_evaluate(fns, env, env_params, *, zero_carry, zero_prev_action, max_steps):
    """Actor-greedy evaluation: mean return, mean steps-to-done, done fraction."""

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
                steps = steps + (1.0 - done)   # steps taken until first done
                done = jnp.maximum(done, d.astype(jnp.float32))
                npa = jnp.where(done > 0, jnp.zeros_like(npa), npa)
                return (nobs, nst, nc, npa, key, ret, done, steps), None

            init = (obs, env_st, carry, prev_action, key,
                    jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
            (_, _, _, _, _, ep_ret, ep_done, ep_steps), _ = jax.lax.scan(
                step_fn, init, length=max_steps)
            return ep_ret, ep_steps, ep_done   # done=1 iff finished within horizon

        keys = jax.random.split(rng_key, n_episodes)
        rets, steps, dones = jax.vmap(run_episode)(keys)
        return jnp.mean(rets), jnp.mean(steps), jnp.mean(dones)

    return _evaluate

def make_evaluate_critic(debug_fns, env, env_params, *, zero_carry, zero_prev_action,
                         max_steps, num_actions, mask_fn=None):
    """Critic-greedy eval: act by argmax_a Q(s,a) over LEGAL actions (via
    debug_fns.predict_qpi), same scan/metrics as the actor-greedy eval.
    Comparing the two localises where a failure is: if critic-greedy plays well
    but actor-greedy doesn't, it's an actor-extraction gap; if both are random,
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
                legalb = (jnp.ones_like(q, dtype=bool) if mask_fn is None
                          else mask_fn(obs) > 0)
                nl = jnp.maximum(jnp.sum(legalb), 1.0)
                # std of Q across LEGAL actions at this state (flatness of the
                # per-action ranking; ≈0 ⇒ Q is a per-step constant).
                qm = jnp.sum(jnp.where(legalb, q, 0.0)) / nl
                qstd = jnp.sqrt(jnp.sum(jnp.where(legalb, (q - qm) ** 2, 0.0)) / nl)
                # range = best-legal − worst-legal Q (the actor's max advantage)
                qrng = (jnp.max(jnp.where(legalb, q, -jnp.inf))
                        - jnp.min(jnp.where(legalb, q, jnp.inf)))
                action = jnp.argmax(jnp.where(legalb, q, -jnp.inf)).astype(jnp.int32)
                npa = jax.nn.one_hot(action, num_actions)   # executed action → next prev-action input
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
                step_fn, init, length=max_steps)
            # per-episode mean over the active (pre-done) steps
            return ep_ret, ep_steps, ep_done, ep_spread / jnp.maximum(ep_steps, 1.0)

        keys = jax.random.split(rng_key, n_episodes)
        rets, steps, dones, spreads = jax.vmap(run_episode)(keys)
        sp = jnp.mean(spreads, axis=0)   # [mean qstd, mean qrange] over actions/episodes
        return jnp.mean(rets), jnp.mean(steps), jnp.mean(dones), sp[0], sp[1]

    return _evaluate

# ── multi-seed training loop ─────────────────────────────────────────────────

def run_seed_group(*, rounds, train_steps, keys, batched, env_state, zero_carry_b,
                   train_v, round_eval, on_round, probe=None, probe_interval=0,
                   qpi=None, qpi_interval=0, tqdm=None, desc=""):
    """Run one group of concurrently-trained seeds for ``rounds`` rounds.

    The async-dispatch overlap is the reason this is shared code: the GPU phase
    that still reads ``batched`` (evaluation, probe-eval collect/train, qpi
    collect) must be pulled to host BEFORE the next round is dispatched, because
    that dispatch donates ``batched`` and ``env_state``.  The pure-CPU work
    (matplotlib rendering) then runs last, overlapping the next round's GPU
    compute.  Getting that order wrong is silently slow, not wrong, so it is
    easy to lose in a copy.

    Callbacks (all owning their own PRNG splits, so key streams stay exact):
      ``train_v(batched, env_state, carry_b, keys) -> (batched, env_state, carry, metrics)``
      ``round_eval(keys, batched) -> (keys, evals)``   evals: name -> per-seed array
      ``on_round(rnd, step, evals, metrics)``          printing and W&B logging
      ``probe``/``qpi``: objects with ``.compute(batched, rnd)`` / ``.render(host)``
    Returns ``(batched, keys)``.
    """
    # Prefetch round 1's training (async); each iteration renders the current
    # round's figures while the NEXT round trains on the GPU.
    keys, train_keys = split_each(keys)
    pending = train_v(batched, env_state, zero_carry_b, train_keys)

    rng = tqdm(range(1, rounds + 1), desc=desc) if tqdm else range(1, rounds + 1)
    for rnd in rng:
        batched, env_state, _ec, metrics = pending
        keys, evals = round_eval(keys, batched)

        # GPU work that still reads `batched` → host BEFORE the next dispatch.
        do_probe = probe is not None and probe_interval > 0 and rnd % probe_interval == 0
        do_qpi = qpi is not None and qpi_interval > 0 and rnd % qpi_interval == 0
        probe_host = probe.compute(batched, rnd) if do_probe else None
        qpi_host = qpi.compute(batched, rnd) if do_qpi else None

        # All reads of batched/env_state are now on host — dispatch the NEXT
        # round's training (async, donates them).
        if rnd < rounds:
            keys, train_keys = split_each(keys)
            pending = train_v(batched, env_state, zero_carry_b, train_keys)

        on_round(rnd, rnd * train_steps, evals, metrics)

        # CPU rendering — overlaps the dispatched round R+1 GPU training.
        if probe_host is not None:
            probe.render(probe_host)
        if qpi_host is not None:
            qpi.render(qpi_host)
    return batched, keys

class Hooks:
    """A (compute → host, render) pair for the periodic probe / heatmap visuals."""

    def __init__(self, compute, render):
        self.compute = compute
        self.render = render
