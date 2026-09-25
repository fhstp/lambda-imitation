# T-maze paper reproduction

This directory retains the numerical experiments used in the paper's main
T-maze figure and appendix. Run commands from the repository root. Exact
experiments need NumPy and, for figures, Matplotlib; neural fits additionally
use the package's JAX/Flax/Optax dependencies.

The tabular model is checked against the bundled Gymnax T-maze. Defaults are
the five-cell corridor (plus a one-cell control), rewards 4 / −0.1, γ=0.9,
and λ=(0,1). Discrepancy is a fixed-coordinate RMS, not a policy-weighted norm.

## Exact fixed points and observation channels

```bash
python examples/tmaze/tmaze_lambda_discrepancy.py
python examples/tmaze/tmaze_retrace_aliasing.py
python examples/tmaze/tmaze_lambda_turnaround.py
```

- `tmaze_lambda_discrepancy.py`: exact Retrace fixed-point solver and
  behaviour/target comparison. Unsupported coordinates are excluded, without
  ridge regularisation.
- `tmaze_retrace_aliasing.py`: interpolate
  `Phi_eta = (1-eta) I + eta Phi_aliased`, for corridor, junction and combined
  aliasing. Intermediate observations are handled by augmenting latent state
  to `(s,o)`. The default six policy pairs include the on-policy 2/3→2/3 and
  off-policy 1/2→2/3 examples in the main figure.
- `tmaze_lambda_turnaround.py`: the appendix's local observation-channel
  control showing that discrepancy need not be monotone in information.

Outputs are JSON and PNG/SVG figures under `tmaze_ld_output/`.
`--output-dir` changes the destination; `--no-figures` writes only numerical
results. For a small execution check:

```bash
python examples/tmaze/tmaze_retrace_aliasing.py --points 5 --no-figures \
  --output-dir outputs/tmaze-smoke
```

## Finite-data detection

The neural experiment fixes policies and one-hot observations, samples complete
forward episodes, and fits four heads (two twins for each λ) using the production
`Head` and `retrace_targets`. It uses detached Huber regression, twin-min target
bootstraps, Adam and EMA. Time weights match the exact evaluator's
discounted-occupancy posterior. No recurrent memory or actor is trained here.

```bash
# Original equal-budget fits: 4,000 updates per dataset size.
python examples/tmaze/tmaze_finite_data_detection.py

# Paper's settled protocol: every dataset gets the same extra 8,000 updates.
python examples/tmaze/tmaze_finite_data_detection.py \
  --settle-updates 8000 --settle-learning-rate 1e-5 \
  --output-dir tmaze_ld_output/finite_data_settled
```

Defaults: five seeds, η in `{0, 0.5, 1}`, 64/256/1024/4096 episodes, batch 128,
hidden widths `(64,64)`, initial learning rate 1e-3, τ=0.005, reward scale 0.1,
and Huber threshold 1. Reported errors and discrepancies are in original reward
units. Each data budget starts from matched fresh initialization and receives
equal optimization. Logs include coverage, regression clipping, signed bias,
centred error and target-network diagnostics.

`--resume` validates protocol and source hashes before continuing a saved run;
`--plot-only` regenerates figures from its JSON. Resume checkpoints from the
pre-publication code are not accepted because the source hashes changed.

## Empirical fixed-point and fitting audit

```bash
python examples/tmaze/tmaze_finite_data_audit.py
python examples/tmaze/tmaze_finite_data_audit.py \
  --input tmaze_ld_output/finite_data_settled/tmaze_finite_data_results.json \
  --output tmaze_ld_output/finite_data_settled_audit.json

# Matched learning-rate continuation controls used by the appendix.
python examples/tmaze/tmaze_finite_data_audit.py --training-checks \
  --output tmaze_ld_output/finite_data_training_audit.json
```

The audit independently constructs empirical Retrace regression moments with
NumPy forward traces, solves their roots and checks them against the production
reverse-scan targets. It separates finite-dataset error from optimization error.
`--training-checks` additionally runs the matched 8,000-step continuation
comparisons for seeds 0, 2 and 4.

The companion paper repository has result snapshots and figure-import scripts.
To verify its exact snapshots against this checkout, point its
`--verify-models` argument to `examples/tmaze`, replacing the old
`examples/lambda-envs` location. Numerical output filenames remain the same.
