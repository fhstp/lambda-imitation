"""Local λ-discrepancy sweeps around an exact endpoint-cancellation policy.

    python examples/tmaze/tmaze_lambda_turnaround.py

The one-cell T-maze has a policy whose on-policy TD(0) and TD(1) values agree
at both endpoints of the observation interpolation Phi_eta = (1-eta)I +
eta Phi_both, yet disagree for 0 < eta < 1. Nearby policies do not generally
return to zero at eta=1, but their discrepancy still turns down before that
endpoint. The lower panels plot d(D^2)/d eta, the directional derivative of
the squared discrepancy objective along this observation-channel path.

This is a diagnostic for a particular representation path, not a claim that a
network's parameter gradient is identical to d/d eta.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import tmaze_lambda_discrepancy as tm
import tmaze_retrace_aliasing as alias


# Exact at gamma=0.9, hall-east=1/5 and junction-up=2/3. The cue-up policy
# waits with the complementary probability via a north no-op.
CENTER_CUE_EAST = float(Fraction(637, 9657))
CENTER_HALL_EAST = 1.0 / 5.0
CENTER_JUNCTION_UP = 2.0 / 3.0
COLORS = ("#56B4E9", "#0072B2", "#111111", "#D55E00", "#E69F00")


def turnaround_policy(cue_up_east=CENTER_CUE_EAST, hall_east=CENTER_HALL_EAST,
                      junction_up=CENTER_JUNCTION_UP):
    """An on-policy cancellation policy and local perturbations of its rows.

    The same hallway row applies to both hidden hallway states, and the same
    junction row applies to both hidden junction states. Therefore it remains a
    valid observation policy at every point of the aliasing interpolation.
    """
    if not all(0 <= value <= 1 for value in (cue_up_east, hall_east, junction_up)):
        raise ValueError("all policy probabilities must lie in [0, 1]")
    policy = tm.junction_policy(junction_up)
    policy[tm.OBS_CUE_UP] = [1 - cue_up_east, 0, cue_up_east, 0]
    policy[tm.OBS_CUE_DOWN] = [0, 0, 1, 0]
    policy[tm.OBS_HALL] = [0, 0, hall_east, 1 - hall_east]
    return policy


def exact_policy_parameters():
    return {"cue_up_east": CENTER_CUE_EAST, "hall_east": CENTER_HALL_EAST,
            "junction_up": CENTER_JUNCTION_UP}


def variants():
    center = exact_policy_parameters()
    return {
        "cue-up east probability": [
            ("−0.020", {**center, "cue_up_east": center["cue_up_east"] - 0.02}),
            ("−0.010", {**center, "cue_up_east": center["cue_up_east"] - 0.01}),
            ("exact", center),
            ("+0.010", {**center, "cue_up_east": center["cue_up_east"] + 0.01}),
            ("+0.020", {**center, "cue_up_east": center["cue_up_east"] + 0.02}),
        ],
        "hallway east probability": [
            ("−0.040", {**center, "hall_east": center["hall_east"] - 0.04}),
            ("−0.020", {**center, "hall_east": center["hall_east"] - 0.02}),
            ("exact", center),
            ("+0.020", {**center, "hall_east": center["hall_east"] + 0.02}),
            ("+0.040", {**center, "hall_east": center["hall_east"] + 0.04}),
        ],
        "junction up probability": [
            ("−0.040", {**center, "junction_up": center["junction_up"] - 0.04}),
            ("−0.020", {**center, "junction_up": center["junction_up"] - 0.02}),
            ("exact", center),
            ("+0.020", {**center, "junction_up": center["junction_up"] + 0.02}),
            ("+0.040", {**center, "junction_up": center["junction_up"] + 0.04}),
        ],
    }


def lambda_curve(pomdp, policy, lambdas, eta_grid, reference_weights):
    """Exact on-policy discrepancy along the Figure-3 both-aliasing channel."""
    state_policy = pomdp.phi @ policy
    values = []
    for eta in eta_grid:
        channel = alias.observation_channel(pomdp, "both", eta)
        # Alias classes share policy rows; stochastic emissions preserve the law.
        if not np.allclose(channel @ state_policy, state_policy, atol=1e-14):
            raise ValueError("policy is not constant on the aliasing classes")
        model = alias.observation_augmented_model(pomdp, channel)
        evaluator = tm.RetraceEvaluator(model, state_policy)
        q_lo, q_hi = tm.value_pair(evaluator, state_policy, lambdas)
        values.append(alias.weighted_gap(q_lo, q_hi, channel.T @ reference_weights))
    values = np.asarray(values)
    loss = values ** 2
    return values, loss, np.gradient(loss, eta_grid, edge_order=2)


def run_sweep(points=401, gamma=0.9, lambdas=(0.0, 1.0)):
    if points < 5:
        raise ValueError("at least five points are required for stable derivatives")
    pomdp = tm.TMazePOMDP(1, gamma=gamma)
    eta = np.linspace(0, 1, points)
    reference = alias.reference_pair_weights(pomdp)
    dimensions = []
    for name, local_variants in variants().items():
        records = []
        for label, params in local_variants:
            policy = turnaround_policy(**params)
            discrepancy, loss, derivative = lambda_curve(
                pomdp, policy, lambdas, eta, reference)
            records.append({"label": label, "parameters": params,
                            "discrepancy": discrepancy.tolist(), "loss": loss.tolist(),
                            "loss_derivative": derivative.tolist(),
                            "endpoint_discrepancy": float(discrepancy[-1]),
                            "peak_discrepancy": float(discrepancy.max()),
                            "peak_eta": float(eta[np.argmax(discrepancy)]),
                            "negative_derivative_fraction": float(np.mean(derivative < -1e-10))})
        dimensions.append({"name": name, "variants": records})
    return {"eta": eta.tolist(), "gamma": gamma, "lambdas": list(lambdas),
            "center_parameters": exact_policy_parameters(),
            "aggregation": "fixed latent-pair RMS weights, transported through Phi_eta",
            "direction": "both-aliasing Phi_eta interpolation", "dimensions": dimensions}


def make_figures(result, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta = np.asarray(result["eta"])
    dimensions = result["dimensions"]
    max_d = max(max(variant["discrepancy"]) for dim in dimensions for variant in dim["variants"])
    max_slope = max(abs(value) for dim in dimensions for variant in dim["variants"]
                    for value in variant["loss_derivative"])
    style = {"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
             "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#333333",
             "xtick.color": "#555555", "ytick.color": "#555555"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, len(dimensions), figsize=(14, 6.2), sharex=True)
        for column, dimension in enumerate(dimensions):
            top, bottom = axes[:, column]
            for color, variant in zip(COLORS, dimension["variants"]):
                width = 2.7 if variant["label"] == "exact" else 1.7
                style = "-" if variant["label"] == "exact" else "-"
                top.plot(eta, variant["discrepancy"], color=color, lw=width, ls=style,
                         label=variant["label"])
                bottom.plot(eta, variant["loss_derivative"], color=color, lw=width, ls=style)
            top.set_title(f"Vary {dimension['name']}", loc="left", pad=10)
            top.set_ylim(-0.025 * max_d, 1.06 * max_d)
            bottom.axhline(0, color="#555555", lw=1)
            bottom.axhspan(-max_slope * 1.06, 0, color="#FDEDEC", zorder=0)
            bottom.annotate("loss decreases as aliasing increases", xy=(0.5, 0.04),
                            xycoords=("axes fraction", "axes fraction"), ha="center",
                            color="#9B2C2C", fontsize=8)
            bottom.set_ylim(-1.06 * max_slope, 1.06 * max_slope)
            for ax in (top, bottom):
                ax.grid(color="#ECECEC", lw=0.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlim(0, 1)
            bottom.set_xlabel("Aliasing mixture η")
        axes[0, 0].set_ylabel("On-policy λ-discrepancy\n(RMS)")
        axes[1, 0].set_ylabel("d(discrepancy²) / dη")
        axes[0, -1].legend(title="Perturbation", fontsize=8, title_fontsize=8,
                            frameon=False, loc="upper right")
        center = result["center_parameters"]
        fig.suptitle("On-policy λ-discrepancy near an endpoint-cancellation policy", fontsize=15, y=0.98)
        fig.text(0.5, 0.012,
                 f"1-cell T-maze · γ = {result['gamma']:g} · λ = ({result['lambdas'][0]:g}, {result['lambdas'][1]:g})"
                 f" · center: cue-east={center['cue_up_east']:.6f}, hall-east=1/5, junction-up=2/3"
                 " · fixed latent-pair RMS weights",
                 ha="center", color="#666666", fontsize=8.5)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.13, hspace=0.14, wspace=0.26)
        stem = output_dir / "tmaze_lambda_turnaround_local"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)

        # Focus on the exact curve's geometry without nearby-policy overlap.
        exact = next(variant for dim in dimensions for variant in dim["variants"]
                     if dim["name"] == "cue-up east probability" and variant["label"] == "exact")
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
        axes[0].plot(eta, exact["discrepancy"], color="#111111", lw=2.4)
        axes[0].fill_between(eta, 0, exact["discrepancy"], color="#0072B2", alpha=0.14)
        axes[0].set_title("Exact policy: zero at both endpoints", loc="left")
        axes[0].set_xlabel("Aliasing mixture η")
        axes[0].set_ylabel("λ-discrepancy (RMS)")
        axes[1].plot(eta, exact["loss_derivative"], color="#111111", lw=2.4)
        axes[1].axhline(0, color="#555555", lw=1)
        axes[1].fill_between(eta, exact["loss_derivative"], 0,
                             where=np.asarray(exact["loss_derivative"]) < 0,
                             color="#D55E00", alpha=0.2)
        axes[1].set_title("Squared-loss directional derivative", loc="left")
        axes[1].set_xlabel("Aliasing mixture η")
        axes[1].set_ylabel("d(discrepancy²) / dη")
        for ax in axes:
            ax.grid(color="#ECECEC", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 1)
        fig.tight_layout()
        stem = output_dir / "tmaze_lambda_turnaround_exact"
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", type=int, default=401)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--lambdas", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--output-dir", type=Path, default=Path("tmaze_ld_output"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.points < 5 or not 0 < args.gamma < 1:
        parser.error("need at least five points and 0 < gamma < 1")
    if not all(0 <= value <= 1 for value in args.lambdas) or args.lambdas[0] == args.lambdas[1]:
        parser.error("choose two distinct lambda values in [0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run_sweep(args.points, args.gamma, args.lambdas)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dimension in result["dimensions"]:
        print(f"\n{dimension['name']}")
        print(f"{'perturbation':>14s} {'D(1)':>10s} {'peak D':>10s} {'peak eta':>10s} {'dL/deta < 0':>14s}")
        for variant in dimension["variants"]:
            print(f"{variant['label']:>14s} {variant['endpoint_discrepancy']:10.6f} "
                  f"{variant['peak_discrepancy']:10.6f} {variant['peak_eta']:10.3f} "
                  f"{variant['negative_derivative_fraction']:14.1%}")
    with (args.output_dir / "tmaze_lambda_turnaround_results.json").open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    if not args.no_figures:
        make_figures(result, args.output_dir)
    print(f"\nResults: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
