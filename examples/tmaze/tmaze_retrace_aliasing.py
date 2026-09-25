"""Figure-3-style observability sweeps for fixed Retrace behaviour/target pairs.

    python examples/tmaze/tmaze_retrace_aliasing.py
    python examples/tmaze/tmaze_retrace_aliasing.py --pairs balanced:2/3 random:1

Following Allen et al. (2024), Figure 3, interpolate Phi_eta = (1-eta) I +
eta Phi_aliased, aliasing the corridor, junction, or both. Policies keep the
same action distribution at each physical state throughout the interpolation.
We retain gamma and the scoring convention of tmaze_lambda_discrepancy.py;
this is an off-policy extension of the figure, not a numerical reproduction.

Intermediate observations are stochastic. We explicitly augment latent states
to (physical state, emitted observation) before using the deterministic-
observation Retrace evaluator. Substituting a stochastic Phi into its lifting
matrix J would incorrectly average the current observation out of the value.
"""

import argparse
from copy import copy
from fractions import Fraction
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

import tmaze_lambda_discrepancy as tm


ALIAS_TYPES = ("corridor", "junction", "both")
COLORS = {"corridor": "#0072B2", "junction": "#D55E00", "both": "#009E73"}
LINESTYLES = {"corridor": "-", "junction": "--", "both": ":"}
DEFAULT_PAIRS = ("paper:2/3", "balanced:1/2", "balanced:2/3",
                 "reverse:2/3", "random:2/3", "random:1")


class PolicyPair(NamedTuple):
    behaviour: str
    p_up: float

    @property
    def target_label(self):
        fraction = Fraction(self.p_up).limit_denominator(1000)
        return str(fraction) if float(fraction) == self.p_up else str(self.p_up)

    @property
    def key(self):
        target = self.target_label.replace("/", "over").replace(".", "p")
        return f"{self.behaviour}_to_{target}"


def aliasing_matrix(pomdp, kind):
    """Square observation channel, with aliased states using one existing label.

    Cue observations and the terminal always remain distinct. Unlike a channel
    that adds separate 'missing' symbols, this is the interpolation in Figure 3.
    """
    if kind not in ALIAS_TYPES:
        raise ValueError(f"unknown aliasing type: {kind}")
    labels = np.arange(pomdp.n_states)
    if kind in ("corridor", "both"):
        labels[np.argmax(pomdp.phi, axis=1) == tm.OBS_HALL] = 1
    if kind in ("junction", "both"):
        labels[np.argmax(pomdp.phi, axis=1) == tm.OBS_JUNC] = pomdp.L + 1
    return np.eye(pomdp.n_states)[labels]


def observation_channel(pomdp, kind, eta):
    if not 0 <= eta <= 1:
        raise ValueError("aliasing mixture must lie in [0, 1]")
    return (1 - eta) * np.eye(pomdp.n_states) + eta * aliasing_matrix(pomdp, kind)


def observation_augmented_model(pomdp, channel):
    """Exact POMDP with deterministic emissions from augmented states (s,o).

    T_aug[(s,o),a,(s',o')] = T[s,a,s'] Phi[s',o']; the initial law is
    p0(s) Phi(s,o). Zero-probability (s,o) states are dropped, without epsilon
    smoothing, including at the two interpolation endpoints.
    """
    if (channel.shape != (pomdp.n_states, pomdp.n_states)
            or np.any(channel < 0) or not np.all(np.isfinite(channel))
            or not np.allclose(channel.sum(axis=1), 1)):
        raise ValueError("channel must be a square row-stochastic matrix")
    terminal_row = np.eye(pomdp.n_states)[pomdp.terminal]
    if not np.array_equal(channel[pomdp.terminal], terminal_row):
        raise ValueError("the terminal must emit its distinct terminal observation")
    states, observations = np.nonzero(channel)
    emission = channel[states, observations]
    model = copy(pomdp)
    model.n_states = len(states)
    model.terminal = int(np.flatnonzero(states == pomdp.terminal)[0])
    model.T = pomdp.T[:, states[:, None], states[None, :]] * emission[None, None, :]
    model.R = pomdp.R[:, states[:, None], states[None, :]]
    model.p0 = pomdp.p0[states] * emission
    model.phi = np.eye(pomdp.n_states)[observations]
    return model


def reference_pair_weights(pomdp):
    """Fixed latent-pair weights that reduce to the earlier five-pair RMS.

    Give each of the five coarse (observation,action) pairs mass 1/5 and
    distribute it over its latent states using a forward-policy occupancy.
    This reference does not depend on the evaluated behaviour or target.
    """
    occupancy = tm.state_occupancy(tm.junction_policy(0.5), pomdp)
    weights = np.zeros((pomdp.n_states, tm.NA))
    for obs, action in tm.SCORE_PAIRS:
        conditional = occupancy * pomdp.phi[:, obs]
        weights[:, action] += conditional / conditional.sum() / len(tm.SCORE_PAIRS)
    return weights


def weighted_gap(q_lo, q_hi, weights):
    """Only score reachable, shared actions; absent observations carry no mass."""
    selected = weights > 0
    gap = (q_hi - q_lo)[selected]
    if not np.all(np.isfinite(gap)):
        raise ValueError("scoring weights include an unsupported observation-action pair")
    return float(np.sqrt(np.sum(weights[selected] * gap ** 2)))


def run_sweep(pomdp, pairs, lambdas, points):
    grid = np.linspace(0, 1, points)
    behaviours = {b.key: b for b in tm.behaviour_policies()}
    # Observation IDs are physical-state IDs at eta=0. Each inherits the same
    # coarse-policy row; revealing the goal never makes either policy use it.
    policies = {key: pomdp.phi @ b.policy for key, b in behaviours.items()}
    targets = {p: pomdp.phi @ tm.junction_policy(p) for p in {pair.p_up for pair in pairs}}
    reference = reference_pair_weights(pomdp)
    records = [{"key": pair.key, "behaviour": pair.behaviour,
                "behaviour_label": behaviours[pair.behaviour].label,
                "target_p_up": pair.p_up, "target_label": pair.target_label,
                "curves": {kind: [] for kind in ALIAS_TYPES}} for pair in pairs]
    for kind in ALIAS_TYPES:
        for eta in grid:
            channel = observation_channel(pomdp, kind, eta)
            model = observation_augmented_model(pomdp, channel)
            weights = channel.T @ reference
            evaluators = {key: tm.RetraceEvaluator(model, policies[key])
                          for key in {pair.behaviour for pair in pairs}}
            for pair, record in zip(pairs, records):
                q_lo, q_hi = tm.value_pair(evaluators[pair.behaviour], targets[pair.p_up], lambdas)
                record["curves"][kind].append(weighted_gap(q_lo, q_hi, weights))
    return {"hallway_length": pomdp.L, "aliasing_grid": grid.tolist(),
            "reference_latent_pair_weights": reference.tolist(), "pairs": records}


def display_order(pomdp):
    """Paper ordering: cue-up/down, paired hallway cells, junctions, terminal."""
    n = pomdp.L + 2
    return np.r_[np.array([[i, n + i] for i in range(n)]).ravel(), pomdp.terminal]


def draw_channel(ax, matrix, pomdp, title, color="#333333"):
    order = display_order(pomdp)
    ax.imshow(matrix[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1,
              interpolation="nearest")
    ticks = np.unique(np.linspace(0, pomdp.n_states - 1, 4).round().astype(int))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7, length=2, pad=2)
    ax.set_xlabel("Observation", fontsize=8, labelpad=3)
    ax.set_ylabel("State", fontsize=8, labelpad=3)
    ax.set_title(title, color=color, fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_curves(ax, record, grid, ymax):
    for kind in ALIAS_TYPES:
        # A tiny floating-point residual is not plotted as a detection signal.
        values = np.array(record["curves"][kind])
        ax.plot(grid, np.where(values < 1e-12, 0, values), color=COLORS[kind],
                ls=LINESTYLES[kind], lw=2.1, label=kind.capitalize())
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.035 * ymax, 1.06 * ymax)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Aliasing mixture η", fontsize=10)
    ax.set_ylabel("Retrace discrepancy (RMS)", fontsize=10)
    ax.grid(color="#ECECEC", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def make_figures(pomdp, result, output_dir, lambdas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    grid, records = result["aliasing_grid"], result["pairs"]
    ymax = max(0.01, max(v for r in records for curve in r["curves"].values() for v in curve))
    handles = [Line2D([0], [0], color=COLORS[k], ls=LINESTYLES[k], lw=2.1,
                      label=k.capitalize()) for k in ALIAS_TYPES]
    caption = (f"{pomdp.L}-cell hallway   ·   γ = {pomdp.gamma:g}   ·   "
               f"λ = ({lambdas[0]:g}, {lambdas[1]:g})   ·   "
               "Φη = (1 − η) Φperfect + η Φaliased   ·   Fixed latent-pair RMS weights")
    style = {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
             "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#333333",
             "xtick.color": "#555555", "ytick.color": "#555555"}
    with plt.rc_context(style):
        for record in records:
            fig, axes = plt.subplots(1, 5, figsize=(14, 3.35),
                                     gridspec_kw={"width_ratios": [1, 3.2, 1, 1, 1]})
            draw_channel(axes[0], np.eye(pomdp.n_states), pomdp, "No aliasing")
            draw_curves(axes[1], record, grid, ymax)
            axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
            for ax, kind in zip(axes[2:], ALIAS_TYPES):
                draw_channel(ax, aliasing_matrix(pomdp, kind), pomdp,
                             f"{kind.capitalize()} aliased", COLORS[kind])
            fig.suptitle(f"Behaviour: {record['behaviour_label']}   →   "
                         f"Target: p(up) = {record['target_label']}", fontsize=13, y=0.98)
            fig.text(0.5, 0.025, caption, ha="center", color="#666666", fontsize=9)
            fig.subplots_adjust(left=0.045, right=0.99, top=0.79, bottom=0.22, wspace=0.6)
            save_figure(fig, output_dir / f"tmaze_retrace_aliasing_L{pomdp.L}_{record['key']}")
            plt.close(fig)

        # Share the observation-matrix key above a compact set of policy pairs.
        cols = min(3, len(records))
        rows = (len(records) + cols - 1) // cols
        height = 2.6 + 2.65 * rows
        fig = plt.figure(figsize=(12.4, height))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 2.4 * rows], hspace=0.35)
        top = outer[0].subgridspec(1, 4, wspace=0.5)
        draw_channel(fig.add_subplot(top[0]), np.eye(pomdp.n_states), pomdp, "No aliasing")
        for i, kind in enumerate(ALIAS_TYPES, start=1):
            draw_channel(fig.add_subplot(top[i]), aliasing_matrix(pomdp, kind), pomdp,
                         f"{kind.capitalize()} aliased", COLORS[kind])
        panels = outer[1].subgridspec(rows, cols, hspace=0.52, wspace=0.34)
        for i, record in enumerate(records):
            ax = fig.add_subplot(panels[i // cols, i % cols])
            draw_curves(ax, record, grid, ymax)
            ax.set_title(f"{record['behaviour_label']} → target {record['target_label']}",
                         loc="left", fontsize=10, pad=10)
        fig.suptitle("How observability changes Retrace discrepancy", fontsize=15,
                     y=1 - 0.12 / height)
        fig.legend(handles=handles, ncol=3, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 1 - 0.38 / height), fontsize=10)
        fig.text(0.5, 0.10 / height, caption, ha="center", fontsize=9, color="#666666")
        fig.subplots_adjust(left=0.075, right=0.985, top=1 - 1.06 / height,
                            bottom=0.75 / height)
        save_figure(fig, output_dir / f"tmaze_retrace_aliasing_L{pomdp.L}_overview")
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hallway-length", type=int, default=5,
                        help="main maze; the one-cell control is also included")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lambdas", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--good-reward", type=float, default=4.0)
    parser.add_argument("--bad-reward", type=float, default=-0.1)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="behaviour:target_p, e.g. paper:2/3 balanced:1/2 random:1")
    parser.add_argument("--output-dir", type=Path, default=Path("tmaze_ld_output"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.hallway_length < 1 or args.points < 2 or not 0 < args.gamma < 1:
        parser.error("need hallway-length >= 1, points >= 2 and 0 < gamma < 1")
    if not all(0 <= l <= 1 for l in args.lambdas) or args.lambdas[0] == args.lambdas[1]:
        parser.error("choose two distinct lambdas in [0, 1]")
    if not np.all(np.isfinite([args.good_reward, args.bad_reward])):
        parser.error("rewards must be finite")
    known = {b.key for b in tm.behaviour_policies()}
    pairs = []
    for value in args.pairs:
        try:
            key, probability = value.split(":")
            p = float(Fraction(probability))
            if key not in known or not 0 <= p <= 1:
                raise ValueError
        except (ValueError, ZeroDivisionError, OverflowError):
            parser.error(f"invalid pair {value!r}; use one of {sorted(known)} followed by :p, 0<=p<=1")
        pair = PolicyPair(key, p)
        if pair not in pairs:
            pairs.append(pair)
    args.pairs = pairs
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for length in dict.fromkeys([args.hallway_length, 1]):
        pomdp = tm.TMazePOMDP(length, args.good_reward, args.bad_reward, args.gamma)
        result = run_sweep(pomdp, args.pairs, args.lambdas, args.points)
        results.append(result)
        print(f"\n{length}-cell hallway: discrepancy at full aliasing (η=1)")
        print(f"{'Behaviour → target':36s} {'Corridor':>10s} {'Junction':>10s} {'Both':>10s}")
        for record in result["pairs"]:
            label = f"{record['behaviour_label']} → {record['target_label']}"
            print(f"{label:36s} " + " ".join(f"{record['curves'][k][-1]:10.6f}" for k in ALIAS_TYPES))
        if not args.no_figures:
            make_figures(pomdp, result, args.output_dir, args.lambdas)
    payload = {"gamma": args.gamma, "lambdas": args.lambdas,
               "good_reward": args.good_reward, "bad_reward": args.bad_reward,
               "observation_channel": "(1-eta)*I + eta*Phi_aliased",
               "aggregation": "fixed reference latent-pair RMS; weights transported through Phi_eta",
               "source_figure": "Allen et al., NeurIPS 2024, Figure 3",
               "experiments": results}
    with (args.output_dir / "tmaze_retrace_aliasing_results.json").open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"\nFigures and numerical results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
