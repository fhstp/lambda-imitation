# Independent off-server recovery kit

In addition to GitHub, the complete repository branch is backed up as a Git
bundle in this **immutable** W&B artifact:

```text
fhstp-data-intelligence-research-group/retrace-rebuttal-validation/b200-recovery-kit-20261115-v1:v0
```

From a new machine with Python 3.12 and access to the same W&B entity:

```bash
python3.12 -m venv restore-env
restore-env/bin/pip install wandb==0.30.0
restore-env/bin/wandb login
restore-env/bin/wandb artifact get --root recovery-kit \
  fhstp-data-intelligence-research-group/retrace-rebuttal-validation/b200-recovery-kit-20261115-v1:v0
cd recovery-kit
sha256sum -c SHA256SUMS
git clone --branch rebuttal/occupancy-correction-2026-11-13 \
  repository.bundle lambda-imitation
cd lambda-imitation
```

Then follow `recovery/b200-2026-11-15/README.md`. The standalone `manifest.json`
and `restore.py` in the kit can also restore the evidence directly without
cloning Git. The final Git bundle contains the compact Git snapshots as well.

The full local evidence (including the companion paper Git history) is stored
separately as:

```text
fhstp-data-intelligence-research-group/retrace-rebuttal-validation/b200-local-evidence-20261115-v1:v0
```

Its exact file SHA-256 and byte size are pinned in `manifest.json`. Original
production artifacts remain in their scientific/coordination projects, at the
exact versioned locations in that manifest. Keep these W&B artifact collections;
deleting them would remove the off-server checkpoint copies. Access credentials
are not backed up and must be supplied separately on the new machine.
