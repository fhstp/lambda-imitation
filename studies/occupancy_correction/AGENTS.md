# Occupancy-correction study — check live admission and state

The user requested a fresh study addressing both discounted-occupancy sampling
and uncorrected / ordinary-IS / Retrace comparisons. This directory is separate
from the completed history assignment inherited by this development worktree.

- Do not launch or reuse the historical B200 queue, tickets, run IDs, or state.
- Do not alter historical sealed source, protocols, evidence, or the manuscript.
- A changed trace coefficient is not a changed starting-state sampler.
- Reject claims of stationary-behaviour theory alignment for mixed adaptive
  replay without a derivation and independent validation.
- Sampling tests alone do not admit production: the actual collection, recurrent
  reconstruction, loss, budgets, and worker lifecycle must also be validated.
- The user authorized proceeding on 2026-11-13. `PROTOCOL.md` and persistent
  `study.json` define the new allocation. Production requires a passing
  `admission.json`, not merely a protocol or successful sampler test.
- Persistent state: `/raid/ustp/experiments/occupancy-correction-2026-11-13`.
  Read `status.json`, per-job records, tmux, GPUs, and W&B before acting.
- Never change admitted source, protocol, seeds, budgets, or run IDs while workers
  are active. Diagnose first; do not reseal a consumed assignment to hide a change.
- Preserve failures. Do not clip ordinary IS, retune, replace seeds, silently
  resume, or retry until a favorable result appears.
- Respect the machine-local dgxB200 host-integrity guardrails. No host driver,
  system package, or service changes are authorized by this study.
