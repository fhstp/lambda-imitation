"""Offline, self-contained HTML return dashboard for a running POPGym suite.

Usage:
    python examples/popgym_report.py --root /tmp/opencode/popgym-weekend
    python examples/popgym_report.py --root /tmp/opencode/popgym-weekend --watch 60

Reads atomic history/status snapshots only. Does not import JAX, contact W&B,
modify training sources, or load executable checkpoints. All charts share a
fixed method palette; validation and final held-out evaluations stay separate.
"""

import argparse
from collections import Counter
from datetime import datetime
import html
import json
import math
from pathlib import Path
import re
import statistics
import time


COLORS = {"lambda": "#D55E00", "sac": "#0072B2", "random": "#6B7280"}
LABELS = {"lambda": "SAC + λ", "sac": "Recurrent SAC", "random": "Random reference"}
MODES = {"greedy": "Greedy actor", "stochastic": "Stochastic actor", "critic": "Critic-greedy"}


def esc(value):
    return html.escape(str(value), quote=True)


def color(method):
    if method in COLORS:
        return COLORS[method]
    # Deterministic fallback for custom runs: never depend on which series is
    # present in a particular panel (unlike plotting-library color cycles).
    import hashlib
    import colorsys
    hue = int(hashlib.sha256(method.encode()).hexdigest()[:8], 16) % 360 / 360
    rgb = colorsys.hls_to_rgb(hue, .38, .65)
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgb)


def stats(values):
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def metric(row, mode):
    try:
        value = float(row["eval"][mode]["return_mean"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def read_json(path, issues, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        issues.append(f"Cannot read {path.name} ({path.parent.name}): {error}")
        return default


def collect(root):
    """Each seed contributes at most once at a given step; no imputed returns."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    environments, issues, counts = {}, [], Counter()
    metadata = {e["env_id"]: e for e in manifest.get("included", [])}
    newest = None
    seen = set()
    for task in manifest["tasks"]:
        env_id, method, seed = task["env_id"], task["method"], task["seed"]
        identity = (env_id, method, seed)
        if identity in seen:
            raise ValueError(f"Duplicate planned run: {identity}")
        seen.add(identity)
        folder = root / env_id / method / f"seed{seed}"
        history_path, status_path = folder / "history.json", folder / "status.json"
        status = read_json(status_path, issues, {})
        if not isinstance(status, dict):
            issues.append(f"Invalid status for {env_id}/{method}/seed{seed}")
            status = {}
        saved_state = status.get("state", "queued")
        target = int(task.get("target_steps", status.get("target_steps", 0)))
        history = read_json(history_path, issues, [])
        execution = read_json(folder / "execution.json", issues, {})
        if not isinstance(history, list):
            issues.append(f"Invalid history for {env_id}/{method}/seed{seed}")
            history = []
        validation, final = {}, None
        for row in history:
            if not isinstance(row, dict):
                continue
            step = row.get("steps")
            if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
                issues.append(f"Invalid training step for {env_id}/{method}/seed{seed}")
                continue
            if row.get("final", False):
                if saved_state == "complete" and step == target:
                    final = row
            else:
                validation[int(step)] = row
        complete = saved_state == "complete" and final is not None
        if saved_state == "complete" and not complete:
            issues.append(f"{env_id}/{method}/seed{seed}: marked complete but no final evaluation at {target} steps")
        counts["complete" if complete else saved_state if saved_state != "complete" else "incomplete"] += 1
        env = environments.setdefault(env_id, {"env_id": env_id, "metadata": metadata.get(env_id, {}),
                                               "methods": {}, "random": None})
        env["methods"].setdefault(method, []).append({"seed": seed, "backend": task.get("backend", "gymnasium"),
                                                      "state": saved_state, "target": target, "complete": complete,
                                                      "validation": validation, "final": final,
                                                      "saved_steps": status.get("steps", max(validation, default=0))})
        env["methods"][method][-1]["execution"] = execution.get("kind", "scalar") if isinstance(execution, dict) else "unknown"
        for path in (history_path, status_path):
            if path.exists():
                newest = max(newest or 0., path.stat().st_mtime)
    for env in environments.values():
        # Random reference is matched across methods, so do not count the same
        # random seed twice. Differing references are flagged rather than pooled.
        refs, conflict = {}, False
        for method in sorted(env["methods"]):
            for run in env["methods"][method]:
                value = metric(run["final"], "random") if run["final"] else None
                if value is not None:
                    if run["seed"] in refs and not math.isclose(refs[run["seed"]], value, rel_tol=1e-6, abs_tol=1e-7):
                        issues.append(f"{env['env_id']}: random references differ across methods for seed{run['seed']}; shared reference omitted")
                        conflict = True
                    refs.setdefault(run["seed"], value)
        if refs and not conflict:
            env["random"] = stats(list(refs.values()))
    return {"environments": environments, "counts": counts, "issues": issues,
             "manifest": manifest, "newest": newest, "planned": len(seen),
            "queue_stop": read_json(root / "queue-stop.json", issues, None),
            "parallel_execution": read_json(root / "parallel-execution.json", issues, None)}


def series(runs, mode):
    at_steps = {}
    for run in runs:
        for step, row in run["validation"].items():
            value = metric(row, mode)
            if value is not None:
                at_steps.setdefault(step, []).append(value)
    return [(step, stats(values)) for step, values in sorted(at_steps.items())]


def finals(runs, mode):
    at_steps = {}
    for run in runs:
        if run["complete"]:
            value = metric(run["final"], mode)
            if value is not None:
                at_steps.setdefault(run["target"], []).append(value)
    return [(step, stats(values)) for step, values in sorted(at_steps.items())]


def env_range(env):
    values = []
    for runs in env["methods"].values():
        for mode in MODES:
            for _, summary in series(runs, mode) + finals(runs, mode):
                sd = summary["sd"] or 0.
                values.extend((summary["mean"] - sd, summary["mean"] + sd))
            for run in runs:
                values.extend(v for row in run["validation"].values() if (v := metric(row, mode)) is not None)
    if env["random"]:
        r = env["random"]
        values.extend((r["mean"] - (r["sd"] or 0), r["mean"] + (r["sd"] or 0)))
    if not values:
        return -1., 1.
    low, high = min(values), max(values)
    padding = max((high - low) * .12, max(abs(low), abs(high)) * .02, .01)
    return low - padding, high + padding


def number(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:.3g}"


def describe(summary):
    sd = f"SD {summary['sd']:.3f}" if summary["sd"] is not None else "SD unavailable"
    return f"mean {summary['mean']:.3f}; {sd}; n={summary['n']} seeds"


def svg_plot(env, mode, xmax):
    """Offline SVG with native hover tooltips and consistent method colors."""
    width, height = 520, 285
    left, right, top, bottom = 65, 15, 18, 54
    pw, ph = width - left - right, height - top - bottom
    low, high = env_range(env)  # common range for all three panels of one task
    x = lambda step: left + step / xmax * pw
    y = lambda value: top + (high - value) / (high - low) * ph
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(env["env_id"])} {esc(MODES[mode])} returns">',
             f'<title>{esc(env["env_id"])} · {esc(MODES[mode])}</title>']
    for i in range(5):
        value = low + i * (high - low) / 4
        yy = y(value)
        parts += [f'<line x1="{left}" x2="{width-right}" y1="{yy:.2f}" y2="{yy:.2f}" class="gridline"/>',
                  f'<text x="{left-8}" y="{yy+4:.2f}" text-anchor="end" class="tick">{esc(number(value))}</text>']
        step = xmax * i / 4
        xx = x(step)
        parts.append(f'<text x="{xx:.2f}" y="{height-bottom+20}" text-anchor="middle" class="tick">{esc(number(step))}</text>')
    parts += [f'<text x="{left+pw/2}" y="{height-8}" text-anchor="middle" class="axis-label">Training interactions</text>',
              f'<text transform="translate(15 {top+ph/2}) rotate(-90)" text-anchor="middle" class="axis-label">Episode return ↑</text>']
    if env["random"]:
        r = env["random"]
        yy = y(r["mean"])
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="{COLORS["random"]}" stroke-width="1.7" stroke-dasharray="7 5"><title>Final random reference: {esc(describe(r))}</title></line>')
    has_data = False
    for method, runs in sorted(env["methods"].items()):
        c = color(method)
        points = series(runs, mode)
        has_data |= bool(points)
        # Individual seed curves are opt-in and never change method hue.
        for run in runs:
            vals = [(step, value) for step, row in sorted(run["validation"].items()) if (value := metric(row, mode)) is not None]
            if vals:
                coords = " ".join(f"{x(s):.2f},{y(v):.2f}" for s, v in vals)
                parts.append(f'<polyline class="seed-trace" points="{coords}" fill="none" stroke="{c}" stroke-width="1"><title>{esc(LABELS.get(method, method))}, seed{esc(run["seed"])}</title></polyline>')
        # Draw SD bands only over contiguous regions with >=2 observed seeds.
        segment = []
        def band(segment):
            if len(segment) < 2:
                return
            upper = [(x(s), y(r["mean"] + r["sd"])) for s, r in segment]
            lower = [(x(s), y(r["mean"] - r["sd"])) for s, r in reversed(segment)]
            coords = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in upper + lower)
            parts.append(f'<polygon points="{coords}" fill="{c}" fill-opacity="0.13"><title>Across-seed sample SD</title></polygon>')
        for point in points:
            if point[1]["sd"] is not None:
                segment.append(point)
            else:
                band(segment); segment = []
        band(segment)
        if points:
            coords = " ".join(f"{x(step):.2f},{y(r['mean']):.2f}" for step, r in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        for step, r in points:
            tip = f"{LABELS.get(method, method)} · validation · {step:,} steps · {describe(r)}"
            parts.append(f'<circle cx="{x(step):.2f}" cy="{y(r["mean"]):.2f}" r="3.5" fill="{c}"><title>{esc(tip)}</title></circle>')
        for step, r in finals(runs, mode):
            has_data = True
            xx, yy = x(step), y(r["mean"])
            if r["sd"] is not None:
                parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{y(r["mean"]+r["sd"]):.2f}" y2="{y(r["mean"]-r["sd"]):.2f}" stroke="{c}" stroke-width="2"/>')
            tip = f"{LABELS.get(method, method)} · FINAL HOLDOUT · {step:,} steps · {describe(r)}"
            parts.append(f'<polygon points="{xx:.2f},{yy-6:.2f} {xx+6:.2f},{yy:.2f} {xx:.2f},{yy+6:.2f} {xx-6:.2f},{yy:.2f}" fill="white" stroke="{c}" stroke-width="2.5"><title>{esc(tip)}</title></polygon>')
    if not has_data:
        parts.append(f'<text x="{left+pw/2}" y="{top+ph/2}" text-anchor="middle" class="empty">No saved return evaluations yet</text>')
    parts.append('</svg>')
    return "".join(parts)


def final_cell(runs, mode):
    values = finals(runs, mode)
    if not values:
        return '<span class="muted">Pending</span>'
    return "<br>".join(f'<span title="{esc(describe(r))}">{r["mean"]:.3f}'
                       + (f' ± {r["sd"]:.3f}' if r["sd"] is not None else '')
                       + f' <small>(n={r["n"]})</small></span>' for _, r in values)


def sort_key(env_id):
    match = re.search(r"(Easy|Medium|Hard)(?:-v\d+)?$", env_id)
    return (env_id[:match.start()], ("Easy", "Medium", "Hard").index(match[1])) if match else (env_id, 3)


def final_comparison(environments):
    """Count task-level wins using all planned seeds and final holdout only."""
    result = {"complete": 0, "policies": {mode: {"wins": 0, "ties": 0, "losses": 0,
                                                "eligible": 0, "skipped": 0} for mode in MODES}}
    for env in environments.values():
        methods = env["methods"]
        runs = [run for rs in methods.values() for run in rs]
        if not runs or not all(run["complete"] for run in runs):
            continue
        result["complete"] += 1
        # No partial-seed comparisons, unequal training budgets, or mixed
        # backends: even a custom manifest must be a matched comparison.
        paired = bool(methods.get("lambda")) and bool(methods.get("sac"))
        if paired:
            settings = lambda rs: {(r["seed"], r["target"], r["backend"]) for r in rs}
            paired = settings(methods["lambda"]) == settings(methods["sac"])
            paired = paired and len({r["target"] for r in runs}) == 1
        for mode, counts in result["policies"].items():
            values = {method: [metric(run["final"], mode) for run in methods.get(method, [])]
                      for method in ("lambda", "sac")}
            if not paired or any(value is None for vs in values.values() for value in vs):
                counts["skipped"] += 1
                continue
            difference = statistics.fmean(values["lambda"]) - statistics.fmean(values["sac"])
            counts["eligible"] += 1
            counts["wins" if difference > 0 else "losses" if difference < 0 else "ties"] += 1
    return result


def comparison_html(environments):
    comparison = final_comparison(environments)
    parts = ['<section id="final-comparison" class="notice"><h2>Final evaluation · λ vs SAC</h2>',
             f'<p>{comparison["complete"]} / {len(environments)} environments have all planned runs finished.</p>',
             '<div class="table-wrap"><table><thead><tr><th>Policy</th><th>λ better / compared</th><th>λ win rate</th><th>Ties</th><th>SAC better</th><th>Skipped</th></tr></thead><tbody>']
    for mode, counts in comparison["policies"].items():
        percentage = f'{100 * counts["wins"] / counts["eligible"]:.1f}%' if counts["eligible"] else '—'
        parts.append(f'<tr><td>{esc(MODES[mode])}</td><td style="color:{COLORS["lambda"]}"><b>{counts["wins"]} / {counts["eligible"]}</b></td>'
                     f'<td>{percentage}</td><td>{counts["ties"]}</td><td style="color:{COLORS["sac"]}">{counts["losses"]}</td><td>{counts["skipped"]}</td></tr>')
    parts += ['</tbody></table></div><div class="muted">Each environment counts once. Compare the unrounded mean final held-out return across all planned seeds; higher is better, exact equality is a tie. Only fully completed environments with matched λ/SAC seeds, budgets and backends and finite final returns for every seed are compared. Skipped counts are among completed environments. This is a descriptive win count, not a significance test, and is independent of the filters below.</div></section>']
    return "".join(parts)


def benchmark_cell(source, env_id):
    """Best within ONE published protocol; never rank across variants."""
    task = source.get("tasks", {}).get(env_id)
    if not task:
        return '<span class="muted">Not reported</span>'
    # Group tied rounded means/SDs so perfect-score ties do not swamp the table.
    groups = {}
    for result in task["best"]:
        groups.setdefault((result["mean"], result["sd"]), []).append(result["model"])
    tip = f'{source["citation"]}; {task["paper_env"]}; best among {task["models_compared"]} models; {source["protocol"]}'
    return '<br>'.join(f'<span title="{esc(tip)}">{esc(" / ".join(models))}<br>'
                      f'<b>{mean:.3f} ± {sd:.3f}</b> <small>(n={source["seeds"]})</small></span>'
                      for (mean, sd), models in groups.items())


def summary_policy_cell(env, mode):
    """Holdout only, with honest partial coverage and matched-cohort emphasis."""
    comparison = final_comparison({env["env_id"]: env})["policies"][mode]
    parts = []
    for method, runs in sorted(env["methods"].items()):
        label = "λ" if method == "lambda" else "SAC" if method == "sac" else method
        better = (method == "lambda" and comparison["wins"] or
                  method == "sac" and comparison["losses"])
        values = finals(runs, mode)
        if not values:
            text = f'{esc(label)}: Pending <small>(0/{len(runs)} seeds)</small>'
        else:
            rows = []
            for step, summary in values:
                tip = f'{LABELS.get(method, method)}; FINAL HOLDOUT; {step:,} training steps; {describe(summary)}'
                text = f'{esc(label)}: {summary["mean"]:.3f}'
                if summary["sd"] is not None:
                    text += f' ± {summary["sd"]:.3f}'
                text += f' <small>({summary["n"]}/{len(runs)} seeds · {number(step)} steps)</small>'
                rows.append(f'<span title="{esc(tip)}">{text}</span>')
            text = '<br>'.join(rows)
        if better:
            text = f'<b>{text}</b>'
        parts.append(f'<div style="color:{color(method)}">{text}</div>')
    return ''.join(parts)


def benchmark_html(environments, benchmarks):
    sources = benchmarks["sources"]
    budgets = sorted({r["target"] for env in environments.values() for rs in env["methods"].values() for r in rs})
    budget_label = " / ".join(number(step) for step in budgets) or "unspecified"
    parts = ['<section id="benchmark-summary" class="notice"><h2>Published benchmarks &amp; our three policies</h2>',
             '<p><b>Reference context, not a matched leaderboard.</b> Published columns show the best method within each cited study and its MMER (best mean episodic return during training), on the original return scale. Our columns use final disjoint holdout returns only; no best-checkpoint selection or validation mixing.</p>',
             f'<p>Published PPO budgets are 15M steps (original DNC: 10M), versus our planned {esc(budget_label)} plus random prefill. Original study: 3 trials; JAX study: 8 seeds. Our cells show mean ± across-seed SD, observed/planned seed coverage and training budget. Partial coverage is provisional, not a five-seed result. <b>Bold local values</b> mark the better λ/SAC method for that policy only when all matched planned runs have finished.</p>',
             '<p>Historical task names/versions and JAX dynamics can differ: JAX references are not verified identical to our audited POPJym ports. StatelessCartPole maps to PositionOnlyCartPole, not VelocityOnlyCartpole. Missing reference results are “Not reported”, never zero. Published ties use rounded table means. “Best” means best in these tables, not a global SOTA claim.</p>',
             '<div class="table-wrap"><table class="benchmark-table"><thead><tr><th>Environment / our backend</th>']
    for source in sources.values():
        parts.append(f'<th>{esc(source["label"])}<br>Best method / MMER</th>')
    parts.extend(f'<th data-policy="{mode}">Our {esc(label)}<br>Final holdout</th>' for mode, label in MODES.items())
    parts.append('</tr></thead><tbody>')
    for env_id, env in sorted(environments.items(), key=lambda item: sort_key(item[0])):
        runs = [r for rs in env["methods"].values() for r in rs]
        backends = sorted({r["backend"] for r in runs})
        state = "complete" if all(r["complete"] for r in runs) else "started" if any(r["validation"] or r["final"] for r in runs) else "pending"
        title = env_id.removeprefix("popgym-").removesuffix("-v0")
        parts.append(f'<tr class="summary-row" data-env="{esc(env_id)}" data-backends="{esc(" ".join(backends))}" data-state="{state}">'
                     f'<td><a href="#{esc(env_id)}">{esc(title)}</a><br><small>{esc(" / ".join(backends))}{" · variant" if "jax" in backends else ""}</small></td>')
        parts.extend(f'<td>{benchmark_cell(source, env_id)}</td>' for source in sources.values())
        parts.extend(f'<td data-policy="{mode}">{summary_policy_cell(env, mode)}</td>' for mode in MODES)
        parts.append('</tr>')
    parts.append('</tbody></table></div><details><summary>Sources and evaluation protocols</summary><ul>')
    for source in sources.values():
        parts.append(f'<li><a href="{esc(source["url"])}" target="_blank" rel="noopener noreferrer">{esc(source["citation"])}</a>: '
                     f'{esc(source["variant"])}. {esc(source["protocol"])}</li>')
    parts.append('</ul><p>PosMLP = positional MLP; Fr.Stack = frame stacking. SD is not a confidence interval; descriptive local bolding is not a significance test. References are bundled locally, with no runtime network requests.</p></details></section>')
    return ''.join(parts)


CSS = """
:root {color-scheme:light; --ink:#17243a; --muted:#65718a; --border:#dce3ed}
* {box-sizing:border-box} body {margin:0;background:#f3f6fb;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header,main,footer {max-width:1680px;margin:auto;padding:24px} h1 {font-size:30px;margin:0 0 5px} h2 {font-size:20px;margin:0} h3 {font-size:15px;margin:0 0 4px}
.subtitle,.muted,small {color:var(--muted)} .kpis,.legend,.toolbar,.badges {display:flex;flex-wrap:wrap;gap:12px;align-items:center}
.kpis {margin:20px 0}.kpi {padding:12px 20px;background:white;border:1px solid var(--border);border-radius:10px}.kpi strong {font-size:24px;display:block}
.swatch {display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px}.legend {gap:22px;margin:12px 0}
.toolbar {background:white;padding:14px;border:1px solid var(--border);border-radius:10px}.toolbar label {font-size:13px} input[type=search],select {padding:7px;border:1px solid var(--border);border-radius:5px;background:white;color:var(--ink)}
.notice {padding:12px 16px;border-left:4px solid #7895b4;background:#eaf0f8;margin-top:16px}.warning {border-color:#be7814;background:#fff3d9}
.card {background:white;border:1px solid var(--border);border-radius:12px;padding:20px;margin:0 0 20px;scroll-margin-top:15px}.card-header {display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:start}
.badge {font-size:12px;background:#edf2f8;border-radius:20px;padding:4px 10px}.coverage {margin:10px 0;font-size:13px}.plots {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.panel svg {display:block;width:100%;height:auto}
.gridline {stroke:#e6ebf2;stroke-width:1}.tick,.axis-label {fill:#65718a;font-size:12px}.empty {fill:#65718a;font-size:13px}
.seed-trace {display:none;opacity:.25} body.show-seeds .seed-trace {display:block} body.single-mode .plots {grid-template-columns:1fr;max-width:850px;margin:auto}
table {border-collapse:collapse;width:100%;font-size:13px;margin:12px 0} th,td {padding:8px 10px;border-bottom:1px solid #edf0f5;text-align:right} th:first-child,td:first-child {text-align:left} th {color:var(--muted);font-weight:500} .table-wrap {overflow:auto}
details {font-size:13px;color:var(--muted)} summary {cursor:pointer;padding:6px 0} .backend-note {font-size:12px;margin-top:4px;max-width:1000px}
.benchmark-table {min-width:1250px;background:white} .benchmark-table td {vertical-align:top} .benchmark-table th {position:sticky;top:0;background:#edf2f8} #benchmark-summary .table-wrap {max-height:650px} .benchmark-table td:first-child {white-space:nowrap} .benchmark-table td:not(:first-child) {min-width:175px}
[hidden] {display:none!important} footer {font-size:13px;color:var(--muted)} @media(max-width:850px) {.plots {grid-template-columns:1fr}.panel {max-width:650px;width:100%;margin:auto}header,main,footer {padding:16px}}
"""

JS = """
const cards = [...document.querySelectorAll('.card')];
const summaryRows = [...document.querySelectorAll('.summary-row')];
function update() {
  const query = document.querySelector('#search').value.trim().toLowerCase();
  const backend = document.querySelector('#backend').value;
  const state = document.querySelector('#state').value;
  const mode = document.querySelector('#mode').value;
  let visible = 0;
  [...cards, ...summaryRows].forEach(card => {
    card.hidden = !card.dataset.env.toLowerCase().includes(query)
      || (backend !== 'all' && !card.dataset.backends.split(' ').includes(backend))
      || (state !== 'all' && card.dataset.state !== state);
    if (!card.hidden && card.classList.contains('card')) visible++;
    card.querySelectorAll('.panel').forEach(panel => panel.hidden = mode !== 'all' && panel.dataset.mode !== mode);
  });
  document.querySelectorAll('[data-policy]').forEach(cell => cell.hidden = mode !== 'all' && cell.dataset.policy !== mode);
  document.body.classList.toggle('single-mode', mode !== 'all');
  document.body.classList.toggle('show-seeds', document.querySelector('#seeds').checked);
  document.querySelector('#visible').textContent = visible + ' / ' + cards.length + ' environments';
}
document.querySelectorAll('.toolbar input,.toolbar select').forEach(control => control.addEventListener('input', update));
update();
"""


def render(snapshot, refresh=0):
    envs, counts = snapshot["environments"], snapshot["counts"]
    xmax = max([1] + [run["target"] for env in envs.values() for runs in env["methods"].values() for run in runs]
               + [s for env in envs.values() for runs in env["methods"].values() for run in runs for s in run["validation"]])
    built = datetime.now().astimezone().isoformat(timespec="seconds")
    newest = datetime.fromtimestamp(snapshot["newest"]).astimezone().isoformat(timespec="seconds") if snapshot["newest"] else "No saved evaluations"
    refresh_tag = f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh else ''
    parts = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">{refresh_tag}',
             f'<title>POPGym · Return dashboard</title><style>{CSS}</style></head><body><header>',
             '<h1>POPGym · Return dashboard</h1><div class="subtitle">Recurrent SAC with and without λ discrepancy · all planned environments</div>',
             f'<div class="subtitle">Generated {esc(built)} · latest saved data {esc(newest)}</div><div class="kpis">',
             f'<div class="kpi"><strong>{counts["complete"]} / {snapshot["planned"]}</strong>Completed runs</div>',
             f'<div class="kpi"><strong>{len(envs)}</strong>Environments</div>',
             f'<div class="kpi"><strong>{sum(any(r["validation"] or r["final"] for rs in e["methods"].values() for r in rs) for e in envs.values())}</strong>With saved returns</div>',
             f'<div class="kpi"><strong>{counts["failed"]}</strong>Saved failures</div></div><div class="legend">']
    methods = sorted({m for e in envs.values() for m in e["methods"]})
    for method in methods + ["random"]:
        parts.append(f'<span><i class="swatch" style="background:{color(method)}"></i>{esc(LABELS.get(method, method))}</span>')
    if snapshot.get("queue_stop"):
        parts.append('<div class="notice warning"><b>Broad queue stopped by request.</b> This page retains the weekend results and last saved checkpoints; unfinished runs are not being continued. New JAX-only temperature diagnostics and the α sweep are separate: <a href="/alpha-sweep/">open the follow-up dashboard</a>.</div>')
    parts += ['</div>', comparison_html(envs), '<div class="toolbar">',
              '<label>Environment <input id="search" type="search" placeholder="e.g. Battleship, Hard"></label>',
              '<label>Backend <select id="backend"><option value="all">All</option><option value="jax">JAX</option><option value="gymnasium">CPU/Gymnasium</option></select></label>',
              '<label>Coverage <select id="state"><option value="all">All</option><option value="started">Partial results</option><option value="complete">All seeds complete</option><option value="pending">No saved returns</option></select></label>',
              '<label>Policy <select id="mode"><option value="all">All policies</option><option value="greedy">Greedy actor</option><option value="stochastic">Stochastic actor</option><option value="critic">Critic-greedy</option></select></label>',
               '<label><input id="seeds" type="checkbox"> Show individual seeds</label><span id="visible" class="muted"></span></div>']
    benchmarks = read_json(Path(__file__).with_name("popgym_benchmarks.json"), snapshot["issues"], None)
    if benchmarks is not None:
        parts.append(benchmark_html(envs, benchmarks))
    parts += [
              '<div class="notice"><b>Read the plots:</b> circles/lines = fixed validation cases; shaded bands = across-seed sample SD (only where n ≥ 2). Hollow diamonds = final disjoint holdout; vertical bars = across-seed SD. Dashed gray = final random-policy reference, counted once per seed. Hover points for exact values and seed counts. Missing data is not filled with zero.</div>',
              '<div class="notice">Partial coverage is not a five-seed result. Seed counts may change along a curve. Y scales are shared across policies within an environment, not across different tasks. Prefill is excluded from training steps. Saved “running” status is not a live process-health check; this page is a snapshot, not a W&B connection.</div>']
    if snapshot["issues"]:
        parts.append('<details class="notice warning"><summary>Data warnings</summary><ul>' + ''.join(f'<li>{esc(i)}</li>' for i in snapshot["issues"]) + '</ul></details>')
    if snapshot.get("parallel_execution"):
        parts.append('<div class="notice"><b>Execution provenance:</b> the scheduler batches unfinished JAX seeds where possible using native-precision vmap; singleton runs and Gymnasium fallbacks remain scalar. Previously completed scalar results are retained. Per-seed execution mode is listed below. Identical floating-point trajectories across modes are not expected. These JAX tasks remain audited POPJym variants, not exact NumPy POPGym replications.</div>')
    if snapshot["manifest"].get("excluded"):
        parts.append('<details><summary>Excluded tasks</summary><ul>' + ''.join(f'<li>{esc(r["env_id"])}: {esc(r["reason"])}</li>' for r in snapshot["manifest"]["excluded"]) + '</ul></details>')
    parts.append('</header><main>')
    for env_id, env in sorted(envs.items(), key=lambda item: sort_key(item[0])):
        runs = [r for rs in env["methods"].values() for r in rs]
        backends = sorted({r["backend"] for r in runs})
        complete = all(r["complete"] for r in runs)
        started = any(r["validation"] or r["final"] for r in runs)
        state = "complete" if complete else "started" if started else "pending"
        parts.append(f'<section id="{esc(env_id)}" class="card" data-env="{esc(env_id)}" data-backends="{esc(" ".join(backends))}" data-state="{state}"><div class="card-header"><div><h2>{esc(env_id)}</h2>')
        parts.append(f'<div class="backend-note muted">{esc(env["metadata"].get("backend_note", ""))}</div></div><div class="badges"><span class="badge">{esc(" / ".join(backends))}</span><span class="badge">{"Complete" if complete else "Partial results" if started else "Not started / awaiting evaluation"}</span></div></div>')
        coverage = []
        for method, rs in sorted(env["methods"].items()):
            observed = sum(bool(r["validation"]) for r in rs)
            done = sum(r["complete"] for r in rs)
            coverage.append(f'{esc(LABELS.get(method, method))}: {observed}/{len(rs)} seeds with curves, {done}/{len(rs)} completed')
        parts.append('<div class="coverage">' + ' · '.join(coverage) + '</div><div class="plots">')
        for mode, label in MODES.items():
            parts.append(f'<div class="panel" data-mode="{mode}"><h3>{label}</h3>{svg_plot(env, mode, xmax)}</div>')
        parts.append('</div><div class="table-wrap"><table><caption>Final holdout returns · mean ± across-seed SD · completed runs only</caption><thead><tr><th>Method</th><th>Seeds complete</th><th>Greedy</th><th>Stochastic</th><th>Critic-greedy</th><th>Random</th></tr></thead><tbody>')
        for method, rs in sorted(env["methods"].items()):
            parts.append(f'<tr><td><i class="swatch" style="background:{color(method)}"></i>{esc(LABELS.get(method, method))}</td><td>{sum(r["complete"] for r in rs)}/{len(rs)}</td>')
            parts.extend(f'<td>{final_cell(rs, mode)}</td>' for mode in (*MODES, "random"))
            parts.append('</tr>')
        parts.append('</tbody></table></div><details><summary>Per-seed saved progress</summary><div class="table-wrap"><table><thead><tr><th>Method / seed</th><th>Execution</th><th>Saved status</th><th>Saved steps</th><th>Target steps</th></tr></thead><tbody>')
        for method, rs in sorted(env["methods"].items()):
            for run in sorted(rs, key=lambda r: r["seed"]):
                parts.append(f'<tr><td>{esc(LABELS.get(method, method))} / {esc(run["seed"])}</td><td>{esc(run.get("execution", "scalar"))}</td><td>{esc(run["state"])}</td><td>{esc(run["saved_steps"])}</td><td>{run["target"]:,}</td></tr>')
        parts.append('</tbody></table></div></details></section>')
    parts += ['</main><footer>Self-contained HTML/SVG: no external assets or network requests. Rebuild with <code>python examples/popgym_report.py --root YOUR_OUTPUT_DIR</code>; add <code>--watch 60</code> for periodic regeneration and browser refresh. Changes to this report do not alter training sources or fingerprints.</footer>',
              f'<script>{JS}</script></body></html>']
    return "".join(parts)


def build(root, output=None, refresh=0):
    root = Path(root)
    output = Path(output) if output else root / "report.html"
    snapshot = collect(root)
    document = render(snapshot, refresh)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    return output, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: ROOT/report.html")
    parser.add_argument("--watch", type=int, default=0, help="regenerate every N seconds and enable browser refresh")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch cannot be negative")
    try:
        while True:
            output, snapshot = build(args.root, args.output, args.watch)
            print(f"Wrote {output}: {len(snapshot['environments'])} environments, {snapshot['counts']['complete']}/{snapshot['planned']} complete", flush=True)
            if not args.watch:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
