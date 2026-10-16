# Read-only Battleship sweep dashboard

Local URL (VPN): **http://10.203.222.228:8000/b200-sweep/**

This independent CPU publisher reads only sweep
`fhstp-data-intelligence-research-group/retrace-discrepancy-battleship-popgym/ekvudzlb`.
It uses the existing report HTTP server; it does not open another port, write to
W&B, load training checkpoints, launch probes or modify training sources/jobs.

## Views

- Equal-checkpoint return comparisons, memory/return scatter, search parameters,
  per-trial return and last16-shot memory trajectories, including failed trials.
- Individual training seeds or mean ± sample seed SD (not a confidence interval).
- Native window/exact-lag curves, decoder/control and evaluation-policy selectors,
  training-time curves/heatmaps, per-seed episode-bootstrap intervals and support.
- Held-out observed-history reconstruction examples, full-history/count probes,
  clipping/Q/gradient diagnostics and actual configuration/provenance.
- Diagnostic warning ledger,rollout/replay parity-error plots and a caution
  notice on memory snapshots whose finite probability-parity check failed.
  Skipped optional measurements appear as gaps,never zero or certified parity.

Missing/undefined metrics stay unavailable, never zero. Mixed-age comparisons
are explicitly marked. Decoder performance does not establish policy use,
unseen-ship inference, causal benefit or a seed-general hyperparameter ranking.

## Operation

Sources: `scripts/battleship_dashboard.py` and `scripts/battleship_dashboard.html`.
The publisher uses W&B Public API credentials already available in the process
environment; no credentials or signed download URLs are published.

```bash
/home/operations/miniforge3/envs/lambda/bin/python scripts/battleship_dashboard.py
```

Add `--watch` for recurring publication. Defaults:

- W&B poll:120 seconds after the preceding cycle; browser poll:30 seconds.
- At most8 artifact imports per cycle, fairly rotating across trials/checkpoints.
- Public output: `/tmp/opencode/popgym-report-web/b200-sweep/`.
- Private cursors/status: `/tmp/opencode/battleship-sweep-dashboard/ekvudzlb/`.
- At least2.5GiB free disk required before importing additional measurements.

History is unsampled and incremental by W&B row ID, preserving sparse rows.
A fresh W&B API client is used for every poll:the SDK's `Api.flush()` clears
run caches but does not invalidate cached sweep objects/their run lists.
Watch mode also uses a fresh one-shot subprocess for each poll,so SDK objects,
JSON parsing heaps and allocator high-water marks cannot accumulate overnight.
The small supervisor retries failed/killed workers and times out a stuck worker
after15 minutes,retaining the last publication and marking refresh health failed.
Only native `result.json`/`reconstructions.json` measurement entries are downloaded,
with SDK caching disabled, into short-lived owned temporary directories. Compact
gzip snapshots retain source SHA256s and native support; checkpoint identity,
complete declared suite, frozen-state and episode-split certificates are checked.
This verifies published certificates, not independent GPU trajectory reproduction.

Last good output survives API/download failures. Stale/error notices distinguish
publisher health from remote training state. Both cache/output are sweep-scoped;
use separate directories for another sweep. The browser lazily loads only the
selected trial's compact snapshots and clears its cache when changing trials.
Chrome/Firefox/Safari versions with native gzip `DecompressionStream` are required.

The previous `jh18ggb5` sweep mirror is preserved as a frozen snapshot at
`http://10.203.222.228:8000/b200-sweep-jh18ggb5/`. Its private cache remains in the
original parent state directory. It is not mixed with the replacement study.

### Local service

`lambda-battleship-sweep-dashboard.service` is a transient user service, independent
of the existing `lambda-popgym-report-http.service`. Only the dashboard publisher
needs restarting after editing its Python/HTML sources. Static reports remain
served when publication is stopped.

```bash
systemctl --user status lambda-battleship-sweep-dashboard.service
systemctl --user restart lambda-battleship-sweep-dashboard.service
```

Publisher logs: `/tmp/opencode/battleship-sweep-dashboard/publisher.log`.
Service limits:1.5GiB RAM,50% of one CPU,64 tasks. Automatic service restart after
failure/OOM is enabled with a15-second delay;worker failures also retry on the
normal polling cadence. The transient service is not reboot-persistent.
To recreate it (only if not already installed):

```bash
systemd-run --user --unit=lambda-battleship-sweep-dashboard.service \
  --description="Read-only W&B B200 sweep return and memory dashboard" \
  --working-directory=/tmp/opencode/battleship-b200-sweep \
  --property=MemoryMax=1610612736 --property=CPUQuota=50% --property=TasksMax=64 \
  --property=Restart=on-failure --property=RestartSec=15 --property=StartLimitIntervalSec=0 \
  --property=StandardOutput=append:/tmp/opencode/battleship-sweep-dashboard/publisher.log \
  --property=StandardError=inherit \
  /home/operations/miniforge3/envs/lambda/bin/python -u \
  scripts/battleship_dashboard.py --watch --interval 120 --artifacts-per-cycle 8
```

## Validation

```bash
/home/operations/miniforge3/envs/lambda/bin/python -m pytest tests/test_battleship_dashboard.py -q
```

`scripts/validate_battleship_dashboard.py` uses Playwright Chromium against the
served page: live plots, selectors/tooltips, responsive/dark layouts, plus isolated
fixtures for failed trials, undefined AUROC, exact-age comparisons and gaps between
unsynchronized checkpoints. Screenshots/proof JSON go to the private
`browser-proof/` directory, not W&B. No training dependencies are imported.
