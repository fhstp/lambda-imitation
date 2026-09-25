"""Compare fixed behaviour policies while varying the Retrace target in T-maze.

Run with just NumPy and Matplotlib (no training or environment package needed)::

    python examples/tmaze/tmaze_lambda_discrepancy.py
    python examples/tmaze/tmaze_lambda_discrepancy.py --lambdas 0.05 0.85

The main experiment uses the reference five-cell maze, gamma=0.9, rewards
4/-0.1, and lambda=(0, 1). A second, one-cell maze removes corridor-position
aliasing: the balanced behaviour then has exactly zero on-policy discrepancy.
Goal-direction aliasing remains. Each comparison holds the environment,
behaviour posterior, and scoring coordinates fixed while varying the target.

We solve the TD-residual Retrace operator in lambda_paper, Appendix
"Discounted-occupancy sampling": W_mu (I-gamma K)^-1 (I-gamma P_pi) J q
= W_mu (I-gamma K)^-1 r, with K built from lambda*min(mu, pi). This is a
two-policy quantity, not an estimate of the target's on-policy discrepancy.
All arithmetic is float64; unsupported coordinates are removed, not regularised.

See tmaze_retrace_discrepancy.md for policy definitions and interpretation.
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


OBS_CUE_UP, OBS_CUE_DOWN, OBS_HALL, OBS_JUNC, OBS_TERM = range(5)
NO, NA = 5, 4
A_NORTH, A_SOUTH, A_EAST, A_WEST = range(4)
OBS_NAMES = ["cue(up)", "cue(down)", "hallway", "junction", "terminal"]
ACTION_NAMES = ["north", "south", "east", "west"]
# All behaviours cover these pairs, and every target acts only on these pairs.
# The norm never changes with either policy. Terminal values are identically 0.
SCORE_PAIRS = ((OBS_CUE_UP, A_EAST), (OBS_CUE_DOWN, A_EAST),
               (OBS_HALL, A_EAST), (OBS_JUNC, A_NORTH), (OBS_JUNC, A_SOUTH))


class TMazePOMDP:
    """Tabular lambda-envs T-maze, with an episodic zero-valued terminal.

    States are (position, goal direction); only the initial observation reveals
    the goal. Positions 1..L share an observation. North/south are no-ops except
    at the junction; east/west move along the corridor and clip at its ends.
    """

    def __init__(self, hallway_length=5, good_reward=4.0, bad_reward=-0.1,
                 gamma=0.9):
        if hallway_length < 1:
            raise ValueError("hallway_length must be at least 1")
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1), as in the paper")
        self.L, self.gamma = hallway_length, gamma
        self.good_reward, self.bad_reward = good_reward, bad_reward
        n_idx = hallway_length + 2
        self.n_states = 2 * n_idx + 1
        self.terminal = self.n_states - 1
        self.T = np.zeros((NA, self.n_states, self.n_states))
        self.R = np.zeros_like(self.T)
        self.phi = np.zeros((self.n_states, NO))
        self.p0 = np.zeros(self.n_states)
        for goal in (0, 1):
            self.p0[goal * n_idx] = 0.5
            for idx in range(n_idx):
                s = goal * n_idx + idx
                obs = goal if idx == 0 else (OBS_JUNC if idx == n_idx - 1 else OBS_HALL)
                self.phi[s, obs] = 1.0
                for action in range(NA):
                    if action in (A_NORTH, A_SOUTH):
                        nxt = self.terminal if idx == n_idx - 1 else s
                        if nxt == self.terminal:
                            self.R[action, s, nxt] = (
                                good_reward if action == goal else bad_reward)
                    else:
                        pos = min(idx + 1, n_idx - 1) if action == A_EAST else max(idx - 1, 0)
                        nxt = goal * n_idx + pos
                    self.T[action, s, nxt] = 1.0
        self.phi[self.terminal, OBS_TERM] = 1.0


def junction_policy(p_up):
    """Walk east, then choose north with probability p_up and south otherwise."""
    if not 0 <= p_up <= 1:
        raise ValueError("junction probability must lie in [0, 1]")
    pi = np.zeros((NO, NA))
    pi[:OBS_JUNC, A_EAST] = 1.0
    pi[OBS_JUNC, [A_NORTH, A_SOUTH]] = [p_up, 1.0 - p_up]
    pi[OBS_TERM] = 1.0 / NA  # unused
    return pi


class Behaviour(NamedTuple):
    key: str
    label: str
    policy: np.ndarray
    color: str
    linestyle: str


def behaviour_policies():
    paper = junction_policy(2.0 / 3.0)
    return (
        Behaviour("paper", "Paper: 2/3 up", paper, "#0072B2", "-"),
        Behaviour("balanced", "Balanced: 1/2 up", junction_policy(0.5), "#009E73", "-"),
        Behaviour("reverse", "Reversed: 1/3 up", junction_policy(1.0 / 3.0), "#D55E00", "-"),
        Behaviour("noisy_paper", "Paper + 50% random", 0.5 * paper + 0.5 / NA,
                  "#CC79A7", "--"),
        Behaviour("random", "Uniform random", np.full((NO, NA), 1.0 / NA),
                  "#8A6D3B", ":"),
    )


def validate_policy(policy, pomdp):
    if (policy.shape != (pomdp.phi.shape[1], NA)
            or not np.all(np.isfinite(policy)) or np.any(policy < 0)
            or not np.allclose(policy.sum(axis=1), 1.0)):
        raise ValueError("policy must have one probability distribution per observation")


def state_occupancy(policy, pomdp):
    """Unnormalised discounted occupancy; its scale cancels in the posterior."""
    validate_policy(policy, pomdp)
    transition = np.einsum("sa,asn->sn", pomdp.phi @ policy, pomdp.T)
    return np.linalg.solve(np.eye(pomdp.n_states) - pomdp.gamma * transition.T,
                           pomdp.p0)


class RetraceEvaluator:
    """Cache the behaviour's posterior W_mu, independently of target and lambda.

    State-action and observation-action vectors use action-major ordering.
    All behaviour-supported nonterminal pairs are solved, including the extra
    actions of exploratory behaviours. Targets must be covered by behaviour.
    """

    def __init__(self, pomdp, behaviour):
        validate_policy(behaviour, pomdp)
        self.pomdp, self.behaviour = pomdp, behaviour
        self.occupancy = state_occupancy(behaviour, pomdp)
        self.obs_occupancy = self.occupancy @ pomdp.phi
        active = self.obs_occupancy > 0
        # Terminal has no outgoing transitions and its value is known, not fitted.
        terminal_obs = np.argmax(pomdp.phi[pomdp.terminal])
        active[terminal_obs] = False
        self.active = active
        self.support = ((behaviour > 0) & active[:, None]).T.ravel()
        self.J = np.kron(np.eye(NA), pomdp.phi)[:, self.support]
        posterior = (self.occupancy[:, None] * pomdp.phi).T
        posterior /= np.where(self.obs_occupancy > 0, self.obs_occupancy, 1)[:, None]
        self.W = np.kron(np.eye(NA), posterior)[self.support]
        self.I = np.eye(NA * pomdp.n_states)
        self.r = (pomdp.T * pomdp.R).sum(axis=-1).ravel()

    def transition(self, action_weights):
        return np.einsum("asn,nb->asbn", self.pomdp.T,
                         self.pomdp.phi @ action_weights).reshape(self.I.shape)

    def system(self, target, lambda_):
        validate_policy(target, self.pomdp)
        if not 0 <= lambda_ <= 1:
            raise ValueError("lambda must lie in [0, 1]")
        if np.any((target > 0) & (self.behaviour == 0) & self.active[:, None]):
            raise ValueError("target actions must be covered by the behaviour policy")
        gamma = self.pomdp.gamma
        K = self.transition(lambda_ * np.minimum(self.behaviour, target))
        P = self.transition(target)
        # Solve rather than explicitly invert the trace resolvent.
        M = np.linalg.solve((self.I - gamma * K).T, self.W.T).T
        return M @ (self.I - gamma * P) @ self.J, M @ self.r

    def q(self, target, lambda_):
        A, rhs = self.system(target, lambda_)
        q = np.linalg.solve(A, rhs)
        if not np.allclose(A @ q, rhs, atol=1e-11, rtol=1e-11):
            raise ArithmeticError("Retrace fixed-point residual is too large")
        # NaNs make unsupported pairs impossible to silently score as zero.
        full = np.full(self.behaviour.T.size, np.nan)
        full[self.support] = q
        full = full.reshape(NA, -1).T
        full[np.argmax(self.pomdp.phi[self.pomdp.terminal])] = 0.0
        return full


def rms_gap(q_lo, q_hi):
    """Fixed RMS norm on the five shared nonterminal observation-action pairs."""
    gaps = np.array([q_hi[o, a] - q_lo[o, a] for o, a in SCORE_PAIRS])
    if not np.all(np.isfinite(gaps)):
        raise ValueError("discrepancy cannot score unsupported values")
    return float(np.sqrt(np.mean(gaps ** 2)))


def value_pair(evaluator, target, lambdas):
    return tuple(evaluator.q(target, l) for l in lambdas)


def discrepancy(evaluator, target, lambdas):
    return rms_gap(*value_pair(evaluator, target, lambdas))


def run_experiment(pomdp, lambdas, points, targets):
    behaviours = behaviour_policies()
    # Include diagonal points and every clipping breakpoint, even on coarse grids.
    knots = [p for b in behaviours for p in
             (b.policy[OBS_JUNC, A_NORTH], 1 - b.policy[OBS_JUNC, A_SOUTH])]
    grid = np.unique(np.r_[np.linspace(0, 1, points), targets, knots])
    records = []
    for behaviour in behaviours:
        evaluator = RetraceEvaluator(pomdp, behaviour.policy)
        own_q = value_pair(evaluator, behaviour.policy, lambdas)
        table = []
        for p_up in targets:
            q_lo, q_hi = value_pair(evaluator, junction_policy(p_up), lambdas)
            table.append({"p_up": p_up, "retrace_discrepancy": rms_gap(q_lo, q_hi),
                          "q_lo": [float(q_lo[o, a]) for o, a in SCORE_PAIRS],
                          "q_hi": [float(q_hi[o, a]) for o, a in SCORE_PAIRS]})
        records.append({
            "key": behaviour.key, "label": behaviour.label,
            "policy": behaviour.policy.tolist(),
            "own_lambda_discrepancy": rms_gap(*own_q),
            "own_supported_max_gap": float(np.nanmax(np.abs(own_q[1] - own_q[0]))),
            "curve": [discrepancy(evaluator, junction_policy(p), lambdas) for p in grid],
            "targets": table,
        })
    on_policy = []
    for p_up in grid:
        pi = junction_policy(p_up)
        evaluator = RetraceEvaluator(pomdp, pi)
        q_lo, q_hi = value_pair(evaluator, pi, lambdas)
        # At p=0 or 1, the unused junction action is not data-identifiable. Its
        # gap is nevertheless known to be zero: that action terminates at once.
        for action in (A_NORTH, A_SOUTH):
            if pi[OBS_JUNC, action] == 0:
                reward = 0.5 * (pomdp.good_reward + pomdp.bad_reward)
                q_lo[OBS_JUNC, action] = q_hi[OBS_JUNC, action] = reward
        on_policy.append(rms_gap(q_lo, q_hi))
    return {"hallway_length": pomdp.L, "target_grid": grid.tolist(),
            "on_policy_curve": on_policy, "behaviours": records}


def make_figures(results, output_dir, gamma, lambdas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    behaviours = behaviour_policies()
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.titlesize": 11, "axes.labelsize": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#333333",
             "xtick.color": "#555555", "ytick.color": "#555555",
             "svg.fonttype": "none", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        height = 2.5 * len(results) + 1.6
        fig, axes = plt.subplots(len(results), 2, figsize=(11.8, height),
                                 squeeze=False, gridspec_kw={"width_ratios": [1, 2.1]})
        for row, result in enumerate(results):
            left, right = axes[row]
            L = result["hallway_length"]
            title = f"{L}-cell hallway"
            title += ": exact-zero control" if L == 1 else (": reference maze" if L == 5 else "")
            left.set_title(f"{'ac'[row]}  Behaviour's own λ-discrepancy", loc="left", pad=12)
            ys = np.arange(len(behaviours))
            for y, record, behaviour in zip(ys, result["behaviours"], behaviours):
                value = record["own_lambda_discrepancy"]
                left.hlines(y, 0, value, color=behaviour.color, lw=3, alpha=0.5)
                left.plot(value, y, "o", color=behaviour.color, ms=6, clip_on=False)
                exact_zero = L == 1 and behaviour.key in ("balanced", "random")
                label = "0 (exact)" if exact_zero else f"{value:.3g}"
                left.annotate(label,
                              (value, y), xytext=(7, 0), textcoords="offset points",
                              va="center", fontsize=9, color=behaviour.color)
            largest = max(r["own_lambda_discrepancy"] for r in result["behaviours"])
            left.set_xlim(0, max(largest * 1.55, 1e-3))
            left.set_yticks(ys, [b.label for b in behaviours])
            left.set_ylim(len(behaviours) - 0.5, -0.5)
            left.set_xlabel("λ-discrepancy (RMS)")
            left.xaxis.set_major_locator(MaxNLocator(3))
            left.grid(axis="x", color="#EEEEEE", lw=0.8)
            left.set_axisbelow(True)

            grid = np.array(result["target_grid"])
            for record, behaviour in zip(result["behaviours"], behaviours):
                right.plot(grid, record["curve"], color=behaviour.color,
                           ls=behaviour.linestyle, lw=2.2, label=behaviour.label)
                right.scatter([v["p_up"] for v in record["targets"]],
                              [v["retrace_discrepancy"] for v in record["targets"]],
                              s=18, facecolor="white", edgecolor=behaviour.color, zorder=4)
            right.plot(grid, result["on_policy_curve"], color="#333333", ls=(0, (4, 3)),
                       lw=1.3, label="On-policy reference (μ = π)", zorder=2)
            right.set_title(f"{'bd'[row]}  {title}", loc="left", pad=12)
            right.set_xlabel("Target probability of choosing up at the junction")
            right.set_ylabel("Retrace discrepancy (RMS)")
            right.set_xlim(0, 1)
            # Keep exact-zero curves visible just above the bottom spine.
            right.set_ylim(bottom=-0.025 * max(right.get_ylim()[1], 1e-3))
            right.set_xticks([0, 1 / 3, 1 / 2, 2 / 3, 1], ["0", "1/3", "1/2", "2/3", "1"])
            right.grid(color="#EEEEEE", lw=0.8)
            right.set_axisbelow(True)
        handles = [Line2D([0], [0], color=b.color, lw=2.2, ls=b.linestyle,
                          label=b.label) for b in behaviours]
        handles.append(Line2D([0], [0], color="#333333", ls="--", lw=1.3,
                              label="On-policy reference (μ = π)"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.005),
                   ncol=3, frameon=False, fontsize=9, columnspacing=2.0)
        fig.text(0.5, 0.10 / height,
                 f"γ = {gamma:g}   ·   λ = ({lambdas[0]:g}, {lambdas[1]:g})   ·   "
                 "Fixed RMS norm over five shared observation–action pairs",
                 ha="center", fontsize=9, color="#666666")
        fig.subplots_adjust(left=0.17, right=0.98, top=1 - 0.86 / height, bottom=0.70 / height,
                            wspace=0.42, hspace=0.53)
        stem = output_dir / "tmaze_retrace_behaviours"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        # A numerical companion: behaviours are rows, fixed targets columns.
        fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 3.5),
                                 squeeze=False)
        vmax = max(v["retrace_discrepancy"] for r in results
                   for b in r["behaviours"] for v in b["targets"])
        for ax, result in zip(axes[0], results):
            values = np.array([[t["retrace_discrepancy"] for t in b["targets"]]
                               for b in result["behaviours"]])
            image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(vmax, 1e-12), aspect="auto")
            for (i, j), v in np.ndenumerate(values):
                text = "0" if v < 1e-12 else f"{v:.3g}"
                ax.text(j, i, text, ha="center", va="center", fontsize=10,
                        color="white" if v > 0.6 * vmax else "#222222")
            ax.set_yticks(range(len(behaviours)), [b.label for b in behaviours])
            ax.set_xticks(range(len(values[0])), [f"{t['p_up']:.3g}" for t in result["behaviours"][0]["targets"]])
            ax.set_xlabel("Target p(up | junction)")
            ax.set_title(f"{result['hallway_length']}-cell hallway", loc="left")
            ax.tick_params(length=0, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout()
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Retrace discrepancy (RMS)",
                     fraction=0.022, pad=0.025)
        stem = output_dir / "tmaze_retrace_matrix"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hallway-length", type=int, default=5)
    parser.add_argument("--good-reward", type=float, default=4.0)
    parser.add_argument("--bad-reward", type=float, default=-0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lambdas", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--targets", type=float, nargs="+", default=(0.0, 1 / 3, 0.5, 2 / 3, 1.0))
    parser.add_argument("--output-dir", type=Path, default=Path("tmaze_ld_output"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.hallway_length < 1 or args.points < 2:
        parser.error("hallway-length must be >= 1 and points must be >= 2")
    if not 0 < args.gamma < 1:
        parser.error("gamma must lie in (0, 1)")
    if not all(0 <= x <= 1 for x in [*args.lambdas, *args.targets]):
        parser.error("lambdas and target probabilities must lie in [0, 1]")
    if args.lambdas[0] == args.lambdas[1]:
        parser.error("choose two distinct lambdas")
    if not np.all(np.isfinite([args.good_reward, args.bad_reward])):
        parser.error("rewards must be finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    lengths = list(dict.fromkeys([args.hallway_length, 1]))
    results = []
    for length in lengths:
        pomdp = TMazePOMDP(length, args.good_reward, args.bad_reward, args.gamma)
        result = run_experiment(pomdp, args.lambdas, args.points, args.targets)
        results.append(result)
        print(f"\nHallway length {length}, gamma={args.gamma}, lambda={tuple(args.lambdas)}")
        print(f"{'Behaviour':25s} {'Own LD':>10s}  " +
              "  ".join(f"π(up)={p:.3g}" for p in args.targets))
        for b in result["behaviours"]:
            print(f"{b['label']:25s} {b['own_lambda_discrepancy']:10.6f}  " +
                  "  ".join(f"{v['retrace_discrepancy']:10.6f}" for v in b["targets"]))
        if length == 1:
            balanced = next(b for b in result["behaviours"] if b["key"] == "balanced")
            assert balanced["own_supported_max_gap"] < 1e-10, "exact-zero control failed"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"gamma": args.gamma, "lambdas": list(args.lambdas),
               "good_reward": args.good_reward, "bad_reward": args.bad_reward,
               "estimator": "exact discounted-occupancy Retrace fixed points",
               "aggregation": "RMS, uniform on five shared nonterminal pairs",
               "score_pairs": [[OBS_NAMES[o], ACTION_NAMES[a]] for o, a in SCORE_PAIRS],
               "observation_order": OBS_NAMES, "action_order": ACTION_NAMES,
               "experiments": results}
    with (args.output_dir / "tmaze_retrace_results.json").open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    if not args.no_figures:
        make_figures(results, args.output_dir, args.gamma, args.lambdas)
    print(f"\nResults written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
