"""Source-host administration ONLY; never interrupts or launches a learner.

Reserve: stop only dispatch, lock the queue, reserve eight unconsumed block-2
jobs, restart dispatch. Sync: import validated B200 results via W&B, no VPN.
"""

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=ROOT/'reviewer_response'
SERVICE='rr-battleship-history-queue.service'
ENTITY='fhstp-data-intelligence-research-group'
VALIDATION_PROJECT='retrace-rebuttal-validation'


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(path.read_text())


def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def restart_dispatch():
    attempt=subprocess.run(['systemctl','--user','start',SERVICE],capture_output=True,text=True)
    if attempt.returncode==0:return
    # A stopped transient systemd-run unit disappears. Recreate its dispatcher,
    # not any learner or tmux session.
    python=ROOT.parent/'lambda-imitation-rebuttal-2026-11-07/.venv/bin/python'
    assert python.exists(),'Original host Python not found; restore dispatch manually'
    subprocess.run(['systemd-run','--user','--unit='+SERVICE.removesuffix('.service'),
        '--property=Restart=on-failure','--property=RestartSec=30','--property=MemoryMax=2G',
        '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=JAX_PLATFORMS=cpu',
        '--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=MPLCONFIGDIR=/tmp/opencode/rebuttal-matplotlib',
        '--working-directory='+str(ROOT),str(python),'-u','-B',str(RUNS/'launch.py'),'queue'],check=True)


def reserve():
    assert (RUNS/'lifecycle').is_dir(),'Run reserve only on the original training host'
    assert not (HERE/'assignment.json').exists(),'Assignment already issued; never regenerate IDs'
    subprocess.run(['systemctl','--user','stop',SERVICE],check=True)
    try:
        with (RUNS/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            jobs=sorted((j for j in read(RUNS/'protocol.json')['jobs'] if j['block']==2),key=lambda j:j['history'])
            assert [j['history'] for j in jobs]==[0,1,2,5,10,15,20,25]
            for job in jobs:
                for suffix in ('.json','.ticket.json'):
                    assert not (RUNS/'lifecycle'/(job['name']+suffix)).exists(),job['name']+' already consumed'
                assert not list(Path(read(RUNS/'protocol.json')['working_root']).glob(
                    f"runs/wandb/run-*-{job['wandb_id']}")),job['name']+' already has W&B output'
            assignment={'schema':1,'lease_id':secrets.token_hex(8),'issued_utc':utc(),
                'issuer':'original RTX Pro 6000 training host','owner':'B200 server, one deployment only',
                'local_dispatch_reserved':True,'automatic_expiry':False,
                'protocol_sha256':sha(RUNS/'protocol.json'),'admission_sha256':sha(RUNS/'admission.json'),
                'entity':ENTITY,'project':'retrace-rebuttal-battleship-history-length-20261109',
                'validation_project':VALIDATION_PROJECT,'jobs':jobs}
            assignment['coordination_run_id']='b200-'+assignment['lease_id']
            # Transaction journal precedes every per-job reservation. Fail-closed
            # if interrupted; never silently return partially reserved jobs.
            write(HERE/'assignment.json',assignment)
            for job in jobs:
                write(RUNS/'lifecycle'/f"{job['name']}.json",{
                    'status':'external-reserved','job':job,'lease_id':assignment['lease_id'],
                    'reserved_utc':utc(),'remote_training_started':False,
                    'note':'Assigned exclusively to B200; no local fallback or automatic expiry'})
            write(HERE/'reservation-complete.json',{
                'lease_id':assignment['lease_id'],'assignment_sha256':sha(HERE/'assignment.json'),
                'jobs':[j['name'] for j in jobs],'local_dispatch_only_was_paused':True,'utc':utc()})
            print(json.dumps(assignment,indent=2))
    except Exception:
        # Once a journal exists, a human must inspect partial reservation before
        # dispatch can restart. Running tmux/scope learners remain unaffected.
        if not (HERE/'assignment.json').exists():
            restart_dispatch()
        raise
    else:restart_dispatch()


def validate_import(directory,job,assignment):
    index=read(directory/'index.json')
    assert index['lease_id']==assignment['lease_id'] and index['job']==job['name']
    assert index['assignment_sha256']==sha(HERE/'assignment.json')
    for name,expected in index['files'].items():
        assert Path(name).name==name,'Invalid bundle path'
        assert sha(directory/name)==expected,name
    receipt=read(directory/'receipt.json')
    assert receipt['job']['wandb_id']==job['wandb_id'] and receipt['job']['flags']==job['flags']
    assert receipt['exit_code']==0 and receipt['continuous_no_resume'] and receipt['all_checkpoints_archived']
    assert receipt['source_admission_sha256']==assignment['admission_sha256']
    assert receipt['deployment_bundle_sha256']==sha(HERE/'bundle-manifest.json')
    config=read(directory/'config.json')
    spec=importlib.util.spec_from_file_location('b200_import_validation',HERE/'worker_queue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate_config(config,job)
    assert config['resume_from'] is None and config['history_length']==job['history']
    assert config['seed']==job['base_seed'] and config['num_seeds']==5
    assert config['rounds']==50 and config['train_steps']==10000 and config['prefill_steps']==12800
    assert config['history_context']==26 and config['eval_episodes']==50
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    assert [r['round'] for r in rows if 'round' in r]==list(range(1,51))
    archives=sorted(directory.glob('checkpoint*.archived.json'))
    assert len(archives)==6 and all(read(p)['remote_committed_verified'] for p in archives)
    return receipt


def sync_once():
    import wandb
    assignment=read(HERE/'assignment.json');api=wandb.Api(timeout=120);imported=[]
    runs=list(api.runs(f"{assignment['entity']}/{assignment['validation_project']}",
                       filters={'name':assignment['coordination_run_id']},per_page=1))
    artifacts=list(runs[0].logged_artifacts()) if runs else []
    for job in assignment['jobs']:
        path=RUNS/'lifecycle'/f"{job['name']}.json"
        current=read(path)
        assert current.get('lease_id')==assignment['lease_id']
        if current['status']=='complete':continue
        versions=[a for a in artifacts if a.type=='history-worker-results' and
                  a.name.split(':')[0]==f"history-b200-result-{job['wandb_id']}"]
        artifact=next((a for a in reversed(versions) if a.state=='COMMITTED'),None)
        if artifact is None:continue
        with tempfile.TemporaryDirectory(prefix='history-b200-import-') as temporary:
            directory=Path(artifact.download(root=temporary,skip_cache=True))
            receipt=validate_import(directory,job,assignment)
            # Independent remote checkpoint-manifest verification, not merely a
            # trust in the worker's boolean or a W&B run's summary/state.
            for pointer in directory.glob('checkpoint*.archived.json'):
                record=read(pointer);remote=api.artifact(record['artifact'])
                entry=remote.manifest.entries[record['name']]
                assert remote.state=='COMMITTED' and entry.digest==record['md5'] and entry.size==record['bytes']
                assert remote.metadata['sha256']==record['sha256']
            for source in directory.iterdir():
                if source.name=='index.json':continue
                destination=RUNS/'records'/f"{job['wandb_id']}.{source.name}"
                if destination.exists():assert sha(destination)==sha(source),'Never overwrite different evidence'
                else:shutil.copyfile(source,destination)
            write(path,{**receipt,'status':'complete','lease_id':assignment['lease_id'],
                'imported_utc':utc(),'result_artifact':artifact.qualified_name,
                'remote_checkpoint_manifests_independently_verified':True,'output':receipt['output']})
            imported.append(job['name'])
    print(json.dumps({'utc':utc(),'imported':imported}),flush=True)
    return all(read(RUNS/'lifecycle'/f"{j['name']}.json")['status']=='complete' for j in assignment['jobs'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('reserve','sync','watch'));args=parser.parse_args()
    if args.action=='reserve':reserve()
    else:
        (HERE/'state').mkdir(exist_ok=True)
        with (HERE/'state/sync.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            while True:
                try:done=sync_once()
                except Exception as error:
                    print(json.dumps({'utc':utc(),'sync_error':repr(error),'training_modified':False}),flush=True)
                    if args.action=='sync':raise
                    done=False
                if args.action=='sync' or done:break
                time.sleep(120)
