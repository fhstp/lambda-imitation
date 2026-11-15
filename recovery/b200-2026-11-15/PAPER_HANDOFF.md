# Continue paper work with only 10 GB free

## Instructions for the next assistant

This is an **archival analysis / manuscript integration task**, not a training
task. Both allocations on the old B200 server are completed. Never run the old
queues, resume checkpoints, retune, or count repeated seeds as independent data.
Read `MIGRATION.md`, this file, and the completion/restoration receipts first.

Branch: `rebuttal/occupancy-correction-2026-11-13` in
`https://github.com/fhstp/lambda-imitation.git`.

The user has only **10 GB free**. Do not install a GPU runtime or download all
checkpoints. No W&B access is needed to use the Git-preserved results.

**Both requested paper datasets are now in Git:** the complete occupancy /
importance-sampling study, and all 16 blocks / 80 seed instances of the history
study. The original host's eight first blocks were additionally recovered from
their immutable final W&B result artifacts. See `PAPER_DATA_COVERAGE.json`.

## Paper-facing numerical tables

- `paper-tables/occupancy-learning.csv.gz`: **27,120 per-seed learning-curve rows**
  for both games and all six sampling/trace conditions, including returns,
  discrepancy measures, importance-ratio and trace-coefficient diagnostics.
- `paper-tables/occupancy-frozen-fits.csv.gz`: **2,340 per-seed fixed-policy fit
  rows** for the separate controls.
- `history-paper/seed_metrics.csv.gz`: **3,840 per-seed history-curve rows** across
  H=0,1,2,5,10,15,20,25, ten seeds each, at all 48 primary reporting checkpoints.
- `history-paper/statistics.json`: history endpoint summaries using the unchanged
  admitted report's Student-t summary function.
- `history-paper/first-blocks/`: original first-host configs, full metrics and
  summaries, plus W&B configurations/source hashes. B200 second-block raw data
  remains under `snapshots/history/`.

These are numerical results, not just figures. They support replotting and
checking both studies without model checkpoints. `paper-tables/README.md`
documents columns and scope. Each supplement has a file-hash index.

Verify the tables on the new server using only Python's standard library:

```bash
python3.12 recovery/b200-2026-11-15/verify_paper_data.py
```

## Small, independent copy of the paper's input data

```bash
python3.12 recovery/b200-2026-11-15/restore.py \
  --destination "$HOME/b200-paper-data" --snapshots-only
```

This materializes approximately **217 MB** of original results/provenance from
approximately 46 MB of compact Git snapshots. It verifies original byte hashes.
These copies remain usable even if W&B's UI or API is unavailable.
The supplemental `history-paper/` and `paper-tables/` data above are already in
the Git checkout; they are additional to these original-server state snapshots.

Important repository files:

- `studies/occupancy_correction/PROTOCOL.md`: exact scientific protocol.
- `snapshots/occupancy/report/summary.json`: final seed values, point estimates,
  pointwise 95% Student-t CIs, exploratory paired contrasts and provenance.
- `snapshots/occupancy/report/*.pdf`: four completed figures.
- `snapshots/occupancy/study.json` and `admission.json`: allocation and source seal.
- `snapshots/occupancy/jobs/`: all job configs, metrics (`.jsonl.gz`), summaries,
  result indices, completion records and immutable artifact receipts.
- `snapshots/history/jobs/*/result/`: the eight B200 history blocks' metrics,
  configurations, receipts and checkpoint indices.
- `COMPLETION_AUDIT.json`, `RESTORE_TEST.json`, `GITHUB_RESTORE_TEST.json`:
  completion and fresh-directory recovery evidence. These are tests on the old
  host, not a claim that another physical server has already been tested.

Snapshot and receipt paths above are relative to `recovery/b200-2026-11-15/`;
the `studies/` protocol path is relative to the repository root.

## Scientific interpretation that must be preserved

- The occupancy study has 36 learning jobs, six shared fixed-dataset jobs and
  36 frozen critic-fit jobs. Ten Battleship / twenty Minesweeper seeds per
  condition; two sampling laws crossed with three trace corrections.
- Adaptive learning uses an empirical mixture of changing collection policies.
  It is **not** an exact stationary-behaviour theorem test. Only the separate
  fixed-mu controls have fixed-policy sampling provenance; neural fits are not
  exact fixed-point solutions or exact Q-error certificates.
- The uniform control uses matched complete-episode reconstruction. It is not
  the legacy contiguous-window sampler. Do not attribute differences against
  historical curves exclusively to discounted occupancy.
- Battleship's primary endpoint is 492,800 interactions including prefill, not
  the later 512,800 training endpoint. Minesweeper's is 1,006,016. No peak picking.
- The original B200 state archive contains only its eight **second** blocks.
  The supplemental `history-paper/` export now recovers the other host's eight
  first blocks too, providing the full predeclared 80 seed instances. Keep these
  origins distinct in provenance; do not count the exports as additional runs.
- First-host results were verified against finished W&B runs, immutable final
  result artifacts, all 50 rounds, configs and admitted source hashes. Original
  first-host lifecycle/exit receipts were not recovered. Do not fabricate those
  receipts or claim they were verified; request them separately if required.
- Incorporate evidence and caveats faithfully. Do not modify the sealed protocol
  or source files to make the narrative or a result more favorable.

## Recover the manuscript and full local evidence, if needed

The companion manuscript's original remote was unreachable. Its files and
complete Git history are in the immutable W&B local-evidence archive.

Using an isolated environment with `wandb==0.30.0`, configure W&B login locally,
then run the following **without `--checkpoints` or `--all-artifacts`**:

```bash
.venv-restore/bin/python recovery/b200-2026-11-15/restore.py \
  --manifest recovery/b200-2026-11-15/manifest.initial.json \
  --destination "$HOME/b200-evidence"
git clone "$HOME/b200-evidence/git/paper.bundle" "$HOME/recovered-paper"
```

This downloads approximately 926 MB and unpacks approximately 1.61 GB. Allow
roughly **3 GB for the files**, plus environment space and safety headroom; the
tool checks available storage before downloading. It includes validation models,
datasets/references and all original logs, but does not fetch the 65 GB of
production artifact uploads. Do not duplicate these directories unnecessarily.

The paper commit is `293c61d4db130f55d3a0b96d64032bf7ac31ef75`. Work on a new
manuscript branch as the user requests; do not overwrite the original snapshot.

## W&B access and preservation

W&B entity: `fhstp-data-intelligence-research-group`.
`manifest.initial.json` pins every original production upload's immutable version,
size and SHA-256. The final `manifest.json`, when present, adds the completed
remote byte audit and publication-bundle index. The restore command checks hashes
even when using the initial index.

The user initially reported a W&B storage-quota warning, then confirmed that it
had cleared with 43 GB available. API retrieval and the full B200 byte audit
succeeded. This is not a guarantee of indefinite future access: keep these W&B
artifacts, and consult the organization admin/W&B support if restrictions return.
Credentials and the old server's SSH key are deliberately not in Git/archives.
