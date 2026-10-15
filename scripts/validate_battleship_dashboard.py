"""Browser proof for the local W&B-only dashboard;requires Playwright Chromium."""
import argparse
import gzip
import json
from pathlib import Path


def fixture():
    config = dict(n_seeds=3, seed_base=42, max_steps=5000000, lambda_coef=10., memory_hidden_dim=256, fe_lr=.0001)
    checkpoints = [dict(step=s, file=f'step{s}.json.gz') for s in (10000, 30000)]
    metric = dict(mean=-.4, std=.02, n=3, seeds={'42': -.42, '43': -.4, '44': -.38})
    points = [dict(step=s, metrics=dict(return_=metric)) for s in (10000, 20000, 30000)]
    for p in points:
        p['metrics'] = {'return': metric, 'diagnostic_seconds': dict(mean=3, std=.2, n=3, seeds={})}
        p['metrics']['parity_max_error'] = dict(mean=1.2638e-5,std=0.,n=3,seeds={'42':1.2638e-5})
    points[1]['metrics'].pop('diagnostic_seconds')
    points[1]['memory_complete'] = False
    run = dict(id='fixture', name='Fixture · measured', state='running', url='https://wandb.ai/fixture', config=config,
               published_step=30000, objective=None, returns=[dict(step=p['step'], mean=-.4, std=.02, n=3) for p in points],
               latest={}, checkpoints=checkpoints, detail='runs/fixture/data.json')
    run['memory_curves'] = {target: [dict(step=p['step'], **metric) for p in points] for target in ('fired', 'hits')}
    run['diagnostic_warnings'] = [dict(steps=10000,seed=42,stage='policy_memory_parity',message='Tolerance exceeded',
        details=dict(mismatched_elements=1,elements=409600,max_probability_error=1.2638e-5)),
        dict(steps=20000,seed=42,stage='probe_fit',message='Optional probe unavailable;measurement omitted',details={})]
    failed = dict(run, id='failed', name='Fixture · failed', state='failed', returns=[], memory_curves={}, checkpoints=[], published_step=10000,diagnostic_warnings=[])
    manifest = dict(sweep='entity/project/test', sweep_url='https://wandb.ai/entity/project/sweeps/test',
                    generated_at='2026-10-15T13:00:00Z', refresh_seconds=120, warnings=[], runs=[run, failed])
    snapshots = {}
    for step in (10000, 30000):
        seeds = {}
        for index, seed in enumerate((42, 43, 44)):
            curves = {}
            for task in ('window', 'lag'):
                for target in ('fired', 'hits'):
                    for mode in ('random', 'stochastic', 'greedy', 'pooled'):
                        for model in ('memory-linear', 'memory-mlp', 'memory-linear-shuffled', 'memory-mlp-shuffled', 'recent-input-mlp', 'prevalence'):
                            curves[f'{task}/{target}/{mode}/{model}'] = [[n, .7+.05*index, .6, .9, 19, 20, 30, 61, .72] for n in (1, 5, 16, 64)]
            board = dict(episode=100, mode=2, age=64, target='fired', labels=[0]*64, scores=[.2]*64, binary=[0]*64)
            seeds[str(seed)] = dict(curves=curves, full_history={}, tasks=128, settings={'probe_updates': 2000}, boards=[board],
                certificate=dict(policy_memory_parity=[dict(passed=False,max_probability_error=1.2638e-5)]))
        snapshots[step] = dict(run_id='fixture', step=step, seeds=seeds, artifact='fixture:v0', fetched_at='2026-10-15T13:00:00Z')
    return manifest, dict(run=run, points=points), snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000/b200-sweep/')
    parser.add_argument('--output', type=Path, default=Path('/tmp/opencode/battleship-sweep-dashboard/browser-proof'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=['--disable-gpu'])
        page = browser.new_page(viewport={'width': 1480, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function('DATA', timeout=120000)
        assert page.evaluate('DATA.sweep.endsWith("/ekvudzlb")')
        assert page.locator('#runs tr').count() == page.evaluate('DATA.runs.length')
        measured = page.evaluate('DATA.runs.find(r=>r.checkpoints.length)?.id')
        if measured:
            page.select_option('#run', measured)
            page.wait_for_function('DETAIL && CURRENT && SNAPSHOTS.length', timeout=120000)
            assert page.locator('#horizons svg').count() == 4
            assert page.locator('#learning svg').count() >= 1
            assert page.locator('#overview-hits svg').count() == 1
            page.select_option('#parameter', 'memory_auroc')
            assert page.locator('#scatter .scatter-point').count() >= 1
            page.locator('#horizon0 .point').first.hover(force=True)
            assert 'eligible' in page.locator('#tooltip').inner_text()
            page.select_option('#seed', '42')
            assert 'episode-bootstrap' in page.locator('#horizons').inner_text()
            page.select_option('#seed', 'all')
            assert page.locator('#boards .board').count() == 4
        page.screenshot(path=str(args.output/'live-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.emulate_media(color_scheme='dark')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(args.output/'live-mobile-dark.png'), full_page=True)

        manifest, detail, snapshots = fixture()
        test = browser.new_page(viewport={'width': 1480, 'height': 1100})
        test.on('pageerror', lambda error: errors.append(str(error)))
        def route(request):
            path = request.request.url.split('?', 1)[0]
            if path.endswith('/health.json'):
                request.fulfill(json=dict(ok=True)); return
            if '/runs/fixture/step' in path:
                step = int(path.split('/step')[-1].split('.')[0])
                request.fulfill(body=gzip.compress(json.dumps(snapshots[step]).encode()), content_type='application/gzip'); return
            if path.endswith('/runs/fixture/data.json'):
                request.fulfill(json=detail); return
            if path.endswith('/runs/failed/data.json'):
                request.fulfill(json=dict(run=manifest['runs'][1], points=[])); return
            request.fulfill(json=manifest)
        test.route('**/data.json?*', route)
        test.route('**/health.json?*', route)
        test.route('**/*.json.gz', route)
        test.goto(args.url)
        test.wait_for_function('CURRENT && SNAPSHOTS.length===2')
        assert test.locator('#runs tr').count() == 2
        assert not test.locator('#diagnostic-warnings').is_hidden()
        assert '409600' in test.locator('#diagnostic-warnings').inner_text()
        assert not test.locator('#parity-warning').is_hidden()
        assert 'NOT certified' in test.locator('#parity-warning').inner_text()
        assert test.locator('#health-parity_max_error svg').count() == 1
        assert test.locator('#boards .board').count() == 4
        test.locator('#horizon0 .point').first.hover(force=True)
        assert 'eligible' in test.locator('#tooltip').inner_text()
        test.select_option('#parameter', 'memory_auroc')
        assert test.locator('#scatter .scatter-point').count() == 1
        # Exact comparison never substitutes a failed run's marker for a return.
        assert '—' in test.locator('#runs tr').nth(1).inner_text()
        # An unsynchronized middle checkpoint must break,not interpolate,a trend.
        paths = test.locator('#trend0 .series-line').evaluate_all('(nodes)=>nodes.map(n=>n.getAttribute("d"))')
        assert paths and all('L' not in d for d in paths), paths
        assert test.locator('#support tr').count() == 48
        test.select_option('#seed', '42')
        assert test.locator('#support tr').count() == 16
        test.select_option('#checkpoint', '10000')
        test.wait_for_function('CURRENT && CURRENT.step===10000')
        test.select_option('#decoder', 'memory-mlp')
        test.uncheck('#controls')
        assert 'Shuffled' not in test.locator('#horizon0 .legend').inner_text()
        test.locator('#n').fill('64'); test.locator('#n').dispatch_event('input')
        assert test.locator('#n-label').inner_text() == '64'
        test.evaluate('Object.values(CURRENT.seeds).forEach(s=>Object.entries(s.curves).forEach(([k,rows])=>{if(k.includes("/hits/"))rows.forEach(r=>{r[1]=null;r[2]=null;r[3]=null;r[4]=0})}));renderHorizons()')
        assert test.locator('#horizon1 .empty').count() == 1
        assert test.locator('#horizon3 .empty').count() == 1
        test.evaluate('refresh()')
        assert test.locator('#seed').input_value() == '42'
        assert test.locator('#checkpoint').input_value() == '10000'
        assert test.locator('#decoder').input_value() == 'memory-mlp'
        test.screenshot(path=str(args.output/'fixture-undefined-hits.png'), full_page=True)
        test.select_option('#run', 'failed')
        test.wait_for_function('DETAIL && DETAIL.run.id==="failed"')
        assert not test.locator('#run-status').is_hidden()
        assert test.locator('#diagnostic-warnings').is_hidden()
        assert test.locator('#parity-warning').is_hidden()
        assert test.locator('#horizons .point').count() == 0
        test.set_viewport_size({'width': 390, 'height': 844})
        assert test.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors, errors
        browser.close()
    proof = dict(passed=True, live_measured_trial=measured,browser_errors=errors, checks=['live replacement sweep identity/counts (may await first measurements)', 'four fixture horizon panels',
        'seed interval/support tooltips', 'return plots', 'four reconstruction maps', 'desktop/mobile/dark',
        'equal-age comparison without marker substitution', 'failed trial visibility', 'missing-checkpoint trend gaps',
        'seed-level support ledger', 'checkpoint/decoder/controls/N64 selectors', 'undefined hits stay unavailable',
        'warning ledger and parity caution without hiding memory/return','skipped optional checkpoints break trends'])
    (args.output/'browser-proof.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
