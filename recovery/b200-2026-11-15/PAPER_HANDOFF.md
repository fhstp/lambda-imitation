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

## Small, independent copy of the paper's input data

```bash
python3.12 recovery/b200-2026-11-15/restore.py \
  --destination "$HOME/b200-paper-data" --snapshots-only
```

This materializes approximately **217 MB** of original results/provenance from
approximately 46 MB of compact Git snapshots. It verifies original byte hashes.
These copies remain usable even if W&B's UI or API is unavailable.

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
- The history archive contains only the eight **second** five-seed blocks from
  this server (40 seed instances). A full 80-seed history-study analysis also
  requires the first blocks from the other host; do not assume they are here or
  inflate the seed count. Keep the original allocation/provenance separate.
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

The user reported a W&B storage-quota warning. API retrieval was still working
during migration, but that is not a guarantee of future access. Never delete these
W&B artifacts to clear the warning. Ask the organization admin/W&B support about
the account's storage/retention restrictions, or arrange an independent backup.
Credentials and the old server's SSH key are deliberately not in Git/archives.
