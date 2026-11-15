# Proposed local cleanup — requires separate user approval

This file is **not deletion authorization**. First confirm final GitHub and W&B
recovery-kit verification. All experiments are completed; do not start/restart
training as part of cleanup. Keep every remote artifact referenced by the
manifest and the GitHub recovery branch.

## Experiment data and migration test copies

Candidate directories, limited to this work:

1. `/raid/ustp/experiments/occupancy-correction-2026-11-13`
2. `/home/ustp/.local/state/lambda-history-b200/32df839bad43f7e9`
3. `/raid/ustp/migration-2026-11-15`

The third directory includes the approximately 66 GB of verified remote downloads,
Git-bundle recovery tests, extraction tests, and migration staging files. They
are local copies, not the off-server backups.

## Source checkouts and project-local runtime

Only if the user also approves removing the local code/manuscript copies:

4. `/raid/ustp/git/lambda-occupancy-correction-2026-11-13`
5. `/raid/ustp/git/lambda-imitation`, including its reproducible `.venv-b200`
6. `/raid/ustp/git/6aa299d71f7b5eb457c918de`

The first two share Git worktree metadata. Handle their worktree relationship
properly, verify no new changes have appeared since backup, and do not delete a
different checkout or another user's data. Move the working session outside a
checkout before removing it.

## Not included automatically

- W&B artifacts or GitHub branches: these **must remain** for recovery.
- The SSH private/public keys `/raid/ustp/git_lamda` and `git_lamda.pub`.
- `/home/ustp/.netrc`, other credentials, OpenCode state/configuration and unrelated
  home-directory files. These were not backed up; arrange separate credential
  cleanup/revocation if required when leaving the server.
- Host drivers, system packages, CUDA installations, services or company data.
- Unrelated tmux sessions/processes. Completed experiment panes may be removed
  separately after confirming their logs are archived and exact session names.

If anything has changed in a candidate directory after the final backup, inspect
and preserve those changes before deleting it. Never turn this narrow plan into
a recursive deletion of `/raid`, `/home/ustp`, or another broad parent directory.
