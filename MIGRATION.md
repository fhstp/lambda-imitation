# Completed B200 experiments: recovery

Use **[the immediate recovery guide](recovery/b200-2026-11-15/RECOVER_NOW.md)** for all code,
results, checkpoints, logs, validation evidence and the companion manuscript.

The branch is `rebuttal/occupancy-correction-2026-11-13`.
Compact results are included in Git. Large artifacts have immutable W&B versions
and byte hashes, with a repository-based restore command. Do not start the old
training queues on the new server: both allocations have already completed.

Full remote download/hash verification has completed. The final `manifest.json`
pins the original local evidence, all 624 B200 production checkpoint files and
all other production/publication entries. See the final recovery guide and
`FINAL_RESTORE_TEST.json`. Local cleanup remains a separate, explicit decision.

With only 10 GB free, use the **[paper-work handoff](recovery/b200-2026-11-15/PAPER_HANDOFF.md)**.
All existing paper statistics, figures and per-seed metrics can be used from Git
without downloading the large checkpoints or accessing W&B.

Both paper additions are covered: the complete occupancy/importance-sampling
comparison and **all 16 blocks / 80 seed instances of the history study**. The
original host's first history blocks were additionally exported from immutable
W&B results. See `recovery/b200-2026-11-15/PAPER_DATA_COVERAGE.json` and the compact
tables linked in the handoff; this does not expand the B200 checkpoint archive's
scope to include the other host's checkpoints.
