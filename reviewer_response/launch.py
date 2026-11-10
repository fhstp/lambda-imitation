"""History-window study admission, tmux worker and non-preemptive GPU queue.

Never resumes or restarts a consumed training attempt. GPU 0--2 remain reserved
until BOTH continuous-control blocks on that GPU are completed and verified.
"""

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREVIOUS=Path('/home/operations/git/lambda-imitation-rebuttal-2026-11-07')
PYTHON=PREVIOUS/'.venv/bin/python'
CONTROLS=PREVIOUS/'reviewer_response/battleship-continuous-2026-11-09'
RAM=Path('/dev/shm/lambda-history-2026-11-09')
PROJECT='retrace-rebuttal-battleship-history-length-20261109'
HISTORIES=(0,1,2,5,10,15,20,25)
sys.path.insert(0,str(HERE))
from protocol import flags


def utc():return datetime.now(timezone.utc).isoformat()


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_hashes():
    paths=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'examples').glob('*.py')),
           *sorted(HERE.glob('*.py')),*sorted((ROOT/'tests').glob('test_history*.py')),HERE/'protocol.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths if p.exists()}


def verify():
    admission=read(HERE/'admission.json');assert admission['passed']
    assert source_hashes()==admission['source_sha256'],'History study sealed source changed'
    import importlib.metadata
    for package,version in admission['runtime'].items():assert importlib.metadata.version(package)==version
    for name,expected in admission['validation_sha256'].items():assert sha(HERE/name)==expected


def prepare():
    assert not (HERE/'protocol.json').exists()
    cases=[read(HERE/f'native-h{h}.json') for h in HISTORIES]
    assert all(c['passed'] and c['validation_only_no_state_reuse'] for c in cases)
    assert len({c['initial_network_sha256'] for c in cases})==len({c['prefill_sha256'] for c in cases})==1
    assert read(HERE/'archive-roundtrip.json')['passed']
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    gpus=[{'index':int(i.strip()),'uuid':u.strip()} for i,u in (row.split(',') for row in gpu_rows.splitlines())]
    assert len(gpus)==6
    jobs=[]
    # Start coverage of the two extremes and an interior H, then fill remaining
    # first blocks. Second blocks are all predeclared, never chosen by results.
    order=[(h,1) for h in (0,10,25,20,15,5,2,1)]+[(h,2) for h in (25,20,15,10,5,2,1,0)]
    for h,block in order:
        seed=(7451234536 if block==1 else 7451234542)
        name=f'battleship-retrace-history{h:02d}-block{block}-5seeds'
        jobs.append({'name':name,'history':h,'block':block,'num_seeds':5,
            'base_seed':seed,'training_seeds':list(range(seed,seed+5)),
            'wandb_id':secrets.token_hex(4),'session':f'rr-history-h{h:02d}-b{block}',
            'flags':flags('battleship','retrace',seed,5)+['--history-length',str(h),
                '--history-context','26','--archive-checkpoints']})
    plan={'date':'2026-11-09','purpose':'Reviewer: controlled memory histories, learned discrepancy and return',
        'base_branch':'publication-cleanup','base_commit':'07381e766181521b3461b87b274970f00d8b46d1',
        'environment':'Battleship 5x5, ships 3,2','method':'AC+Retrace with MSE auxiliary regression',
        'histories':list(HISTORIES),'training_seeds_per_history':10,'seed_blocks':[5,5],
        'history_definition':'H preceding network inputs x_(t-H)..x_(t-1) plus current x_t; x_t=(latest shot outcome,previous action). GRU starts at zero separately for EVERY decision window. H=0 is one-step GRU, no recurrent history.',
        'capacity_control':'Identical 512-unit GRU, embedding, heads, parameter count and common-seed initialization for every H, including H=0. Actor gradients into feature extractor stopped at every H.',
        'episode_boundaries':'Left padding is invalid and does not advance GRU. Raw histories and previous action reset at episode boundaries and at the original paper round boundaries.',
        'mask_caveat':'The full paper legal-action mask is unchanged and encodes earlier shot locations; H caps network representation history, NOT all information available to the policy.',
        'replay_context':'26 extra raw context-only transitions at EVERY H. Original B=10, learning L=40, tail=50 retained; actor uses 90 positions, task/discrepancy 89, auxiliary regression 40. Prefix recovers previous-action inputs even for H=25. Common sampler window=126 rather than historical 100; prefill FIXED at original 12800.',
        'budgets':'50 rounds of 10000 online steps plus 12800 random prefill. Same 48 evaluation checkpoints through 492800 total interactions enter the <=500000 paper window. No tuning.',
        'evaluation_episodes':50,'analysis_interaction_limit':500000,'primary_endpoint_interactions':492800,
        'primary_return':'Mean greedy return per seed, pointwise 95% Student-t across ten seeds',
        'primary_discrepancy':'At each checkpoint on independent sampled-policy evaluation episodes: sqrt(mean over visited transitions of sum_a pi_current(a|window)*[min Q_lambda1_online - min Q_lambda2_online]^2). Each seed RMS is computed before averaging seeds.',
        'secondary_discrepancy':'Replay training RMS sqrt(train/ld_loss/seed_i); greedy-trajectory policy-weighted RMS; online auxiliary twin disagreement RMS and regression losses as fitting diagnostics.',
        'interpretation_limits':['Learned finite-data critic disagreement, not an exact Retrace fixed-point norm.',
            'Policies and visitation differ across trained H. This is a controlled end-to-end representation ablation, NOT a fixed-policy monotonicity theorem test.',
            'No promise of monotone discrepancy or return; zero learned discrepancy alone is not a sufficiency certificate.',
            'Identical numerical seed labels across H support exploratory pairing, not additional independent seeds. No pooling with previous studies.'],
        'failure_policy':'No resume, seed replacement, clipping, retuning, dropped failures or automatic training retry. Administrative error pauses dispatch for inspection; no active job preemption.',
        'storage':'W&B-managed RAM run directories. Every original-cadence checkpoint uploaded with cache skipped; COMMITTED remote MD5/size/SHA metadata verified BEFORE releasing its own local staging copy. Failure checkpoint files are retained. Small metrics/config/archive receipts mirrored to persistent repository storage every queue tick. No old-study files touched.',
        'scheduler':'One training process per tmux session, only its assigned GPU visible. Fill physically free GPUs, but reserve control GPUs until all original blocks complete, including between-block handoffs. Check RAM/disk before every dispatch.',
        'working_root':str(RAM),'gpus':gpus,'jobs':jobs}
    write(HERE/'protocol.json',plan)
    for name in ('lifecycle','records','results'):(HERE/name).mkdir(exist_ok=True)
    for name in ('runs','cache','data','tmp'):(RAM/name).mkdir(parents=True,exist_ok=True)
    import importlib.metadata
    import xml.etree.ElementTree as ET
    test_xml=ET.parse(HERE/'full-cpu-tests.xml').getroot()
    suites=list(test_xml.iter('testsuite'))
    assert suites and all(int(s.attrib.get('failures',0))==int(s.attrib.get('errors',0))==0 for s in suites)
    admission={'passed':True,'utc':utc(),'source_sha256':source_hashes(),
        'runtime':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','flax','optax','numpy','wandb')},
        'validation_sha256':{n:sha(HERE/n) for n in ('full-cpu-tests.xml','history-cpu-tests.xml',
            'queue-cpu-tests.xml','archive-roundtrip.json',*[f'native-h{h}.json' for h in HISTORIES])},
        'native_scope':'64 updates at production dimensions per tested H; no claim of deterministic long-horizon replay',
        'production_training_state_reused':False}
    write(HERE/'source_manifest.json',admission['source_sha256']);write(HERE/'admission.json',admission)
    print(json.dumps({'jobs':len(jobs),'histories':list(HISTORIES),'admission_passed':True}))


def environment(job):
    env=dict(os.environ);env.pop('WANDB_SWEEP_ID',None)
    env.update(CUDA_VISIBLE_DEVICES=job['gpu_uuid'],JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONPATH=str(ROOT/'src'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_ENTITY='fhstp-data-intelligence-research-group',
        WANDB_RUN_ID=job['wandb_id'],WANDB_RESUME='never',WANDB_MODE='online',
        WANDB_RUN_GROUP=f"battleship-retrace-history-{job['history']:02d}",
        WANDB_JOB_TYPE='controlled-history-learning',WANDB_TAGS=f"history-window,h{job['history']},retrace,mse,10-matched-seeds,no-resume",
        WANDB_DIR=str(RAM/'runs'),WANDB_CACHE_DIR=str(RAM/'cache'),WANDB_DATA_DIR=str(RAM/'data'),
        TMPDIR=str(RAM/'tmp'),REBUTTAL_SOURCE_MANIFEST=str(HERE/'source_manifest.json'),
        REBUTTAL_CHECKPOINT_BYTES=str(550000000))
    return env


def gpu_occupied():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {line.split(',')[0].strip() for line in text.splitlines()}


def controls_reserved():
    reserved=set()
    for job in read(CONTROLS/'protocol.json')['jobs']:
        r=read(CONTROLS/'lifecycle'/f"{job['name']}.json",{})
        if r.get('status')!='complete' or not r.get('all_checkpoints_committed_verified'):
            reserved.add(job['gpu_uuid'])
    return reserved


def available_memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024


def resource_ok(active):
    # Native full-shape host RSS was ~3GiB; admit 10GiB per active/new worker,
    # separately reserve all remaining 44GiB forecast of the continuous study.
    old_used=int(subprocess.check_output(['du','-s','-B1','/dev/shm/lambda-battleship-continuous-2026-11-09'],text=True).split()[0])
    future_old=max(0,44*2**30-old_used) if controls_reserved() else 0
    return (available_memory()>future_old+(active+1)*10*2**30+10*2**30
        and shutil.disk_usage('/dev/shm').free>future_old+8*2**30
        and shutil.disk_usage(ROOT).free>3*2**30)


def spawn(job,gpu):
    verify()
    assert gpu['uuid'] not in gpu_occupied()|controls_reserved()
    assignment={**job,'gpu_uuid':gpu['uuid'],'gpu_index':gpu['index']}
    with (HERE/'lifecycle'/f"{job['name']}.ticket.json").open('x') as stream:json.dump(assignment,stream,indent=2)
    subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT)],check=True)
    subprocess.run(['tmux','set-option','-t',job['session'],'remain-on-exit','on'],check=True)
    env=environment(assignment)
    keys=['WANDB_ENTITY','WANDB_RUN_ID','WANDB_RESUME','WANDB_MODE','WANDB_RUN_GROUP',
        'WANDB_JOB_TYPE','WANDB_TAGS','WANDB_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR',
        'REBUTTAL_SOURCE_MANIFEST','REBUTTAL_CHECKPOINT_BYTES','CUDA_VISIBLE_DEVICES','JAX_PLATFORMS',
        'XLA_PYTHON_CLIENT_PREALLOCATE','PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TMPDIR']
    # Explicit full required environment, independent of stale tmux-server env.
    command=shlex.join(['env','-u','WANDB_SWEEP_ID',*[f'{k}={env[k]}' for k in keys],
                       str(PYTHON),'-u','-B',str(HERE/'launch.py'),'run','--job',job['name']])
    subprocess.run(['tmux','send-keys','-t',job['session'],'exec '+command,'Enter'],check=True)
    print(json.dumps({'utc':utc(),'launched':job['name'],'gpu':gpu['index'],'wandb_id':job['wandb_id']}),flush=True)


def output_for(job):
    paths=list((RAM/'runs/wandb').glob(f"run-*-{job['wandb_id']}/files/training"))
    assert len(paths)<=1
    return paths[0] if paths else None


def mirror(job):
    output=output_for(job)
    if output:
        for path in output.iterdir():
            if path.suffix not in ('.json','.jsonl'):continue
            dest=HERE/'records'/f"{job['wandb_id']}.{path.name}"
            tmp=dest.with_suffix(dest.suffix+'.tmp')
            shutil.copyfile(path,tmp);tmp.replace(dest)


def run_job(name):
    verify();job=read(HERE/'lifecycle'/f'{name}.ticket.json')
    assert job and os.environ['CUDA_VISIBLE_DEVICES']==job['gpu_uuid']
    receipt=HERE/'lifecycle'/f'{name}.json'
    assert not receipt.exists(),'Never restart a consumed history attempt'
    assert '--resume-from' not in job['flags']
    record={'status':'running','started_utc':utc(),'job':job,'parent_pid':os.getpid(),'resume_allowed':False}
    write(receipt,record)
    command=['systemd-run','--user','--scope','--quiet',f"--unit=rr-history-{job['wandb_id']}",
        '-p','MemoryHigh=8G','-p','MemoryMax=10G','-p','MemorySwapMax=0',str(PYTHON),'-u','-B',
        'examples/battleship.py',*job['flags'],'--wandb','--wandb-project',PROJECT,'--wandb-run-name',name]
    with (HERE/'lifecycle'/f'{name}.console.txt').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment(job),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        record.update(command=command,process_pid=process.pid);write(receipt,record)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
    output=output_for(job);mirror(job)
    record.update(status='complete' if code==0 else 'failed',exit_code=code,
                  finished_utc=utc(),output=str(output) if output else None)
    if code==0:
        config=read(output/'config.json');assert config['resume_from'] is None
        rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
        assert [r['round'] for r in rows if 'round' in r]==list(range(1,51))
        archives=list(output.glob('checkpoint*.archived.json'));assert len(archives)==6
        assert all(read(p)['remote_committed_verified'] for p in archives)
        record.update(continuous_no_resume=True,all_checkpoints_archived=True)
    write(receipt,record)
    if code==0:subprocess.run(['tmux','kill-session','-t',job['session']])
    return code


def notify(event):
    with (HERE/'alerts.jsonl').open('a') as stream:stream.write(json.dumps({'utc':utc(),**event})+'\n')
    print(json.dumps(event),flush=True)
    subprocess.run(['tmux','wait-for','-S','rr-training-monitor-alert'],timeout=10)


def once(dispatch=True):
    verify();plan=read(HERE/'protocol.json');states={};events=[];active_gpu=set();active=0
    panes=subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name}\t#{pane_dead}'],text=True)
    panes=dict(line.split('\t') for line in panes.splitlines())
    for job in plan['jobs']:
        mirror(job)
        receipt=read(HERE/'lifecycle'/f"{job['name']}.json",{})
        ticket=read(HERE/'lifecycle'/f"{job['name']}.ticket.json")
        state=receipt.get('status','starting' if ticket else 'queued')
        output=output_for(job);last={}
        if output and (output/'metrics.jsonl').exists():
            for line in (output/'metrics.jsonl').read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if 'round' in r:last=r
        states[job['name']]={'status':state,'round':last.get('round'),
            'history':job['history'],'gpu':ticket['gpu_index'] if ticket else None,
            'return':last.get('eval/greedy/return/mean'),
            'sampled_discrepancy_rms':last.get('eval/sampled/discrepancy_rms/mean')}
        if state in ('starting','running'):
            active+=1;active_gpu.add(ticket['gpu_uuid'])
            if panes.get(job['session'])!='0':events.append({'key':'worker-lost:'+job['name']})
            age=time.time()-(output/'metrics.jsonl').stat().st_mtime if output and (output/'metrics.jsonl').exists() else 0
            expected=read(HERE/'native-h25.json')['steady_seconds_per_update']*10000
            if age>max(7200,expected*4+900):events.append({'key':'stalled:'+job['name'],'age_seconds':age})
        if state in ('failed','administrative-error'):events.append({'key':'failed:'+job['name']})
    occupied=gpu_occupied()|controls_reserved()|active_gpu
    if dispatch and not events:
        for gpu in plan['gpus']:
            if gpu['uuid'] in occupied or not resource_ok(active):continue
            job=next((j for j in plan['jobs'] if states[j['name']]['status']=='queued'),None)
            if job is None:break
            spawn(job,gpu);active+=1;occupied.add(gpu['uuid']);states[job['name']].update(status='starting',gpu=gpu['index'])
    complete=all(s['status']=='complete' for s in states.values())
    if complete:
        report=subprocess.run([str(PYTHON),'-B',str(HERE/'report.py')],capture_output=True,text=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'},cwd=ROOT)
        events.append({'key':'all-complete' if report.returncode==0 else 'final-report-failed',
                       'stdout':report.stdout[-2000:],'stderr':report.stderr[-2000:]})
    old=read(HERE/'alert-state.json',{})
    for event in events:
        if event['key'] not in old:notify(event)
    write(HERE/'alert-state.json',{e['key']:e for e in events})
    write(HERE/'status.json',{'utc':utc(),'complete':complete,'jobs':states,'active_alerts':events,
        'continuous_control_gpu_reservations':sorted(controls_reserved()),'host_available_bytes':available_memory()})
    return complete


def publish():
    verify()
    import wandb
    run=wandb.init(entity='fhstp-data-intelligence-research-group',project='retrace-rebuttal-validation',
        name='battleship-controlled-history-admission',job_type='history-study-admission',
        config={'production_training':False,'histories':list(HISTORIES),'seeds_per_history':10})
    artifact=wandb.Artifact(f'battleship-history-admission-{run.id}',type='source')
    for name in read(HERE/'admission.json')['source_sha256']:
        artifact.add_file(str(ROOT/name),name=name,policy='immutable',skip_cache=True)
    for name in ('admission.json','full-cpu-tests.xml','queue-cpu-tests.xml','history-cpu-tests.xml',
                 *[f'native-h{h}.json' for h in HISTORIES],'archive-roundtrip.json','import-provenance.json'):
        artifact.add_file(str(HERE/name),name='reviewer_response/'+name,policy='immutable',skip_cache=True)
    uploaded=run.log_artifact(artifact);uploaded.wait();assert uploaded.state=='COMMITTED'
    run.finish();write(HERE/'publication.json',{'artifact':uploaded.qualified_name,'utc':utc()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','publish','queue','once','run'));parser.add_argument('--job')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='publish':publish()
    elif args.action=='run':
        try:sys.exit(run_job(args.job))
        except Exception as error:
            path=HERE/'lifecycle'/f'{args.job}.json';r=read(path,{})
            r.update(status='administrative-error',error=repr(error),traceback=traceback.format_exc(),
                     training_retry_allowed=False);write(path,r);raise
    else:
        assert (HERE/'publication.json').exists()
        with (HERE/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                done=once()
                if args.action=='once' or done:break
                time.sleep(30)
