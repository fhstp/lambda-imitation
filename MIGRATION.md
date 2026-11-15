# Completed B200 experiments: recovery

Use **[the immediate recovery guide](recovery/b200-2026-11-15/RECOVER_NOW.md)** for all code,
results, checkpoints, logs, validation evidence and the companion manuscript.

The branch is `rebuttal/occupancy-correction-2026-11-13`.
Compact results are included in Git. Large artifacts have immutable W&B versions
and byte hashes, with a repository-based restore command. Do not start the old
training queues on the new server: both allocations have already completed.

Full re-download verification is still in progress. You can restore on the new
server now, but do not delete the originals until recovery is verified and
cleanup is explicitly agreed.

With only 10 GB free, use the **[paper-work handoff](recovery/b200-2026-11-15/PAPER_HANDOFF.md)**.
All existing paper statistics, figures and per-seed metrics can be used from Git
without downloading the large checkpoints or accessing W&B.
