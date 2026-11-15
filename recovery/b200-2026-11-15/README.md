# Recover all work from the B200 server

This branch preserves both completed studies performed on `dgxB200`:

- **Occupancy correction:** 36 learning blocks, six dataset blocks, 36 frozen
  critic-fit blocks; 576 production checkpoints, six frozen references and six
  fixed datasets. All 78 jobs completed, with every planned seed retained.
- **History lengths:** the eight reserved **second** five-seed blocks, 40 seed
  instances, with 48 checkpoints. The first blocks ran on another host and are
  not claimed as part of this server backup. Do not rerun either allocation.

For paper integration, `history-paper/` additionally preserves the original
host's first-block numerical results: the full 80-seed history dataset is now
available in Git. `paper-tables/` also contains tidy per-seed tables for the
complete importance-sampling/occupancy study. See `PAPER_HANDOFF.md`. This does
not mean the original host's large checkpoints were included in this server's
65 GB byte-audit scope.

Code, protocols, compact metrics, results, figures and provenance are in Git.
Large checkpoints and the complete local evidence archive are in W&B, indexed
by **immutable versions, SHA-256, MD5 and byte size** in `manifest.json`.
W&B access to `fhstp-data-intelligence-research-group` is required for those bytes;
a plain Git clone does not download the checkpoints. This is archival recovery,
**not authorization to resume or repeat training**.

## 1. Clone this branch

```bash
git clone --branch rebuttal/occupancy-correction-2026-11-13 \
  https://github.com/fhstp/lambda-imitation.git
cd lambda-imitation
```

If GitHub is unavailable, the independent W&B recovery kit documented in
`REMOTE.md` contains a complete Git bundle of this branch and the restore files.

## 2. Results only, directly from Git

No Python packages, W&B, GPU or network are needed for this step (Python 3.12):

```bash
python3.12 recovery/b200-2026-11-15/restore.py \
  --destination /your/storage/b200-recovered --snapshots-only
```

The `snapshots/` directory also contains readable per-job configurations,
summaries, provenance and upload receipts; metrics are losslessly gzip-compressed.
Final occupancy statistics and four PDF figures are under
`snapshots/occupancy/report/`. Metrics are verified against original byte hashes
during restoration, without changing scientific files or recalculating results.

## 3. Restore everything, including all checkpoints

Allow **at least 80 GB free storage**. Use an isolated download environment:

```bash
python3.12 -m venv .venv-restore
.venv-restore/bin/pip install wandb==0.30.0
.venv-restore/bin/wandb login
.venv-restore/bin/python recovery/b200-2026-11-15/restore.py \
  --destination /your/storage/b200-recovered --checkpoints --all-artifacts
```

Without `--checkpoints`, the complete local evidence archive is still restored,
including retained references/datasets and validation checkpoints. With
`--checkpoints`, all 870 production upload entries are restored (624 checkpoint
files plus configurations/results/references/datasets). `--all-artifacts` also
restores the published source, per-job bundles and report artifacts.
The command verifies every download and every original local file. It may be
repeated: identical files are retained, conflicting files are never overwritten.
It never launches a learner, imports JAX, or loads a pickle.

Restored layout:

```text
occupancy-state/       # complete original state, validation, logs, datasets, results
history-state/         # complete B200 history state and production checkpoints
occupancy-checkout/    # original checkout files at migration time
history-checkout/      # original history checkout, including local-only audit notes
paper-checkout/        # complete companion manuscript files
git/                  # both repositories as complete Git bundles
runtime/              # exact installed package versions and Python/platform record
publications/         # independently recovered published artifact contents
local-inventory.json  # original-file SHA-256 and sizes
exclusions.json       # explicit omissions and nonessential symlink aliases
```

The original absolute paths in provenance are historical records, not paths that
must exist on the new machine. Use the restored state directory explicitly:

```bash
# Requires the scientific Python dependencies; writes only regenerated reports.
PYTHONPATH=src:. python -m studies.occupancy_correction.report \
  --state /your/storage/b200-recovered/occupancy-state
```

## Runtime, manuscript and exclusions

- Scientific runtime: Python 3.12, JAX/JAXlib 0.6.2, Flax 0.10.7, Optax 0.2.5,
  NumPy 1.26.4. `deployment/b200/requirements.txt` records the admitted stack;
  `runtime/runtime.json` records all actually installed package versions.
  Use a new isolated environment if execution is needed. Do **not** run the
  completed deployment queue or modify host drivers/CUDA to restore files.
- Checkpoint objects are trusted study-owned pickle/gzip-pickle files. Hash
  verification does not make arbitrary third-party pickle safe. Use the matching
  code and dependencies when inspecting these objects; no automatic unpickling
  or policy resumption is part of this recovery tool.
- The companion paper remote was unreachable during migration. Its files and
  complete Git history are independently archived. To make a working checkout:
  `git clone /your/storage/b200-recovered/git/paper.bundle recovered-paper`.
  The recorded paper commit is `293c61d4db130f55d3a0b96d64032bf7ac31ef75`.
- Installed virtual environments, regenerable Python/test caches, redundant W&B
  symlink aliases, credentials, shell history, machine-global configuration and
  OpenCode databases are excluded. Original regular-file logs, offline W&B run
  data, validation failures/earlier receipts and local-only scientific notes are
  preserved. No credentials or host-global safety rules were copied into Git.

`manifest.json` records the remote byte-verification outcome. `RESTORE_TEST.json`
records an independent fresh-directory extraction and hash verification.
`FINAL_RESTORE_TEST.json` records a complete restore-tool verification of all
2,332 indexed remote files, including every B200 production checkpoint, and all
4,210 original local regular files. The tests ran on this host, not another
physical server. `CLEANUP_PLAN.md` lists the narrow proposed removal scope.
No local deletion is authorized by these records; cleanup is a separate decision.
