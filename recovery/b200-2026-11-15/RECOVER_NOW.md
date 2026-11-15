# Restore on the next server now

The full byte-by-byte re-download audit on the old server is **still in progress**.
Do not delete the original files until recovery is verified and cleanup is agreed.
This initial index nevertheless includes all original local evidence and all
870 production upload entries, so restoration on the new server can start now.
The final index will additionally catalogue redundant published result bundles.

For the paper, both numerical datasets are available directly in Git, including
the subsequently recovered original-host first history blocks: **all 80 history
seed instances**, plus the complete importance-sampling/occupancy comparisons.
See `PAPER_HANDOFF.md`, `PAPER_DATA_COVERAGE.json`, `paper-tables/`, and
`history-paper/`. These supplements do not require downloading all checkpoints.

## Complete data restore

On the **new server**, allow at least **80 GB free storage**:

```bash
git clone --branch rebuttal/occupancy-correction-2026-11-13 \
  https://github.com/fhstp/lambda-imitation.git
cd lambda-imitation
python3.12 -m venv .venv-restore
.venv-restore/bin/pip install wandb==0.30.0
.venv-restore/bin/wandb login
.venv-restore/bin/python recovery/b200-2026-11-15/restore.py \
  --manifest recovery/b200-2026-11-15/manifest.initial.json \
  --destination /your/storage/b200-recovered --checkpoints
```

Use your actual storage directory instead of `/your/storage/b200-recovered`.
Your W&B account needs access to `fhstp-data-intelligence-research-group`.

Successful completion prints:

```text
Restoration verified. No training was started; no pickle was loaded.
```

This success is conditional on verifying **every** selected download's byte size
and SHA-256, plus every original archived file. Failures stop the command. It
never silently skips missing files or overwrites conflicting existing files.

Included:

- All **624 saved production checkpoints**: 576 occupancy and 48 B200 history.
- All **4,210 original archived regular files**: study definitions, seeds,
  configurations, metrics, summaries, figures, logs, offline W&B files,
  validation evidence and validation checkpoints, retained datasets/references,
  both source checkouts, local audit notes, and the companion manuscript.
- Complete Git bundles for the code and companion paper, plus installed package
  version records. The paper's own remote need not be available.

Excluded deliberately: credentials (including the SSH private key), installed
venvs, regenerable Python/test caches, redundant symlink aliases, machine-global
configuration and OpenCode databases. No first-block history training from the
other host is claimed as part of this server's backup.

## Results without W&B

The compact metrics, figures, summaries, configuration and provenance are also
in Git. To materialize those without installing any packages:

```bash
python3.12 recovery/b200-2026-11-15/restore.py \
  --destination /your/storage/b200-results --snapshots-only
```

`COMPLETION_AUDIT.json` confirms all 86 jobs finished locally and in W&B.
`RESTORE_TEST.json` records a fresh-directory extraction of all 4,210 files,
successful clones of both Git bundles, and exact reproduction of the scientific
report except its timestamp. These tests ran in fresh directories on the old
server, **not on another physical server**. Run the command above on your new
server for that final independent confirmation.
