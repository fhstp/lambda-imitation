# Numerical data for both paper additions

No W&B login or model checkpoints are required to use these Git-preserved data.
All values are extracted from original per-seed metrics, not digitized plots or
run-level averages. Raw metrics, configurations and provenance are also retained.

## Importance sampling / discounted occupancy

`occupancy-learning.csv.gz` contains 27,120 rows:

- Environment, starting measure, trace correction, block and actual training seed.
- Round and actual environment interactions including prefill.
- Greedy return, sampled-policy discrepancy RMS, sampled-policy discounted
  discrepancy RMS and greedy-trajectory discrepancy RMS.
- Logged importance-ratio maximum, trace-coefficient maximum and anchor age.

`primary_reporting_window` identifies the prespecified reporting range. The full
raw training curve remains present; do not select a favorable later endpoint.
Training diagnostics are blank at initial evaluation, when no updates occurred.

`occupancy-frozen-fits.csv.gz` contains 2,340 rows, with per-seed held-out fixed-mu
discrepancy at every declared update checkpoint, under discounted and uniform
starting-state weighting, and auxiliary twin disagreement. These controls are
separate from the adaptive learning comparison.

The exact original metric-key mapping and input hashes are in
`../PAPER_DATA_COVERAGE.json`. The scientific protocol is
`../../../studies/occupancy_correction/PROTOCOL.md` from this directory.

## History as controlled memory

`../history-paper/seed_metrics.csv.gz` contains 3,840 rows: eight history lengths,
ten seeds per history, and 48 reporting checkpoints through 492,800 interactions.
Columns are history length, actual seed, interactions, greedy return,
sampled-policy discrepancy RMS, greedy-trajectory discrepancy RMS, auxiliary twin
RMS and replay discrepancy RMS. The replay RMS is the square root of the original
per-seed discrepancy loss, matching the admitted reporting code.

The eight original-host first blocks were downloaded from their immutable final
W&B result artifacts and validated against exact run IDs, grids, source hashes
and protocol configs. Their original process-lifecycle receipts are not included;
this limitation is explicit in `../history-paper/provenance.json`.

## Important distinction

These tables are sufficient to reproduce the **existing numerical comparisons,
statistics and figures**. A new model evaluation, rollout or representation probe
may require fetching selected checkpoints, plus separate authorization to run
that new work. Those saved models remain separately indexed; they are not
replaced by the CSV exports.
