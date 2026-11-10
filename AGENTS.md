# Active scientific experiments — read before acting

Read `deployment/b200/AGENT_HANDOFF.md` and `deployment/b200/README.md` first.
The short startup prompt is `deployment/b200/AGENT_PROMPT.txt`.

Scientific source, protocol, seed IDs, and validation files are hash sealed.
Do not edit them, upgrade dependencies, pull updates into active workers, start
duplicate runs, resume checkpoints, or silently change estimators/budgets.
First inspect live state using `bash deployment/b200/run.sh status`, tmux,
GPU processes and W&B. Read-only diagnosis is safe. No subagents are required.

This branch supplies the reserved B200 **second** five-seed blocks only. It is
not an instruction to rerun the original host's first blocks or other studies.
