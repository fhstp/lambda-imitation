"""Portable B200 execution of ONLY the eight explicitly reserved seed blocks.

No OpenCode, VPN client, local-source-host path, systemd, distributed gradient
training, checkpoint resume, or automatic learner retry is required or used.
"""

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=ROOT/'reviewer_response'


def utc():return datetime.now(timezone.utc).isoformat()


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:raise RuntimeError(message)


def assignment():
    a=read(HERE/'assignment.json')
    check(a is not None and a['local_dispatch_reserved'],'No issued local reservation')
    check(sha(HERE/'assignment.json')==read(HERE/'reservation-complete.json')['assignment_sha256'],
          'Incomplete or changed reservation')
    check(sha(RUNS/'protocol.json')==a['protocol_sha256'],'Changed scientific protocol')
    check(sha(RUNS/'admission.json')==a['admission_sha256'],'Changed scientific admission')
    canonical={j['name']:j for j in read(RUNS/'protocol.json')['jobs']}
    check([j['history'] for j in a['jobs']]==[0,1,2,5,10,15,20,25],'Invalid history assignment')
    for j in a['jobs']:
        check(canonical[j['name']]==j and j['block']==2 and j['num_seeds']==5,'Changed seed block')
        check('--resume-from' not in j['flags'],'Checkpoint resume is forbidden')
    return a


def verify(runtime=True):
    a=assignment();admission=read(RUNS/'admission.json')
    for name,digest in admission['source_sha256'].items():
        check(sha(ROOT/name)==digest,'Changed sealed scientific source: '+name)
    for name,digest in admission['validation_sha256'].items():
        check(sha(RUNS/name)==digest,'Changed original validation: '+name)
    bundle=read(HERE/'bundle-manifest.json')
    check(bundle is not None,'Deployment bundle has not been sealed')
    for name,digest in bundle['sha256'].items():
        check(sha(ROOT/name)==digest,'Changed deployment bundle: '+name)
    if runtime:
        check(sys.version_info[:2]==(3,12),'Use Python 3.12 and setup.sh, not a system environment')
        for name,version in admission['runtime'].items():
            check(importlib.metadata.version(name)==version,'Runtime mismatch: '+name)
        for line in (HERE/'requirements.txt').read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            package,version=line.split('==');package=package.split('[')[0]
            check(importlib.metadata.version(package)==version,'Pinned package mismatch: '+package)
    return a


def state_path(value=None):
    return Path(value or os.environ.get('HISTORY_B200_STATE') or
        Path.home()/'.local/state/lambda-history-b200'/assignment()['lease_id']).expanduser().resolve()


def owner():
    machine=Path('/etc/machine-id').read_text().strip() if Path('/etc/machine-id').exists() else socket.gethostname()
    return hashlib.sha256((socket.gethostname()+'|'+machine).encode()).hexdigest()


def gpu_list():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version',
                                 '--format=csv,noheader,nounits'],text=True)
    result=[]
    for line in text.splitlines():
        i,u,n,m,d=[x.strip() for x in line.split(',')]
        result.append({'index':int(i),'uuid':u,'name':n,'memory_total_mib':int(m),'driver_version':d})
    return result


def occupied():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {line.split(',')[0].strip() for line in text.splitlines()}


def select_gpus(spec,gpus):
    indices=[int(s) for s in spec.split(',')]
    check(len(indices)==len(set(indices))==8,'Select exactly eight distinct full B200 GPU indices')
    by_index={g['index']:g for g in gpus}
    check(set(indices)<=set(by_index),'Requested GPU index not found')
    selected=[by_index[i] for i in indices]
    check(all('B200' in g['name'].upper() and g['memory_total_mib']>=90000 for g in selected),
          'Production is restricted to eight full B200 cards, not the original host or MIG slices')
    return selected


def environment(job,config,gpu=None):
    env=dict(os.environ);env.pop('WANDB_SWEEP_ID',None)
    work=Path(config['work'])
    env.update(PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES=(gpu or job['gpu'])['uuid'],JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_ENTITY=assignment()['entity'],
        WANDB_MODE='online',WANDB_RUN_ID=job['wandb_id'],WANDB_RESUME='never',
        WANDB_RUN_GROUP=f"battleship-retrace-history-{job['history']:02d}",
        WANDB_JOB_TYPE='controlled-history-learning-b200',
        WANDB_TAGS=f"history-window,h{job['history']},retrace,mse,10-matched-seeds,no-resume,B200,block2",
        WANDB_NOTES='Reserved second five-seed block, fresh policy/replay/optimizer initialization. B200 native validation state is discarded.',
        WANDB_DIR=str(work/'runs'),WANDB_CACHE_DIR=str(work/'cache'),WANDB_DATA_DIR=str(work/'data'),
        TMPDIR=str(work/'tmp'),REBUTTAL_SOURCE_MANIFEST=str(RUNS/'source_manifest.json'),
        REBUTTAL_CHECKPOINT_BYTES='550000000')
    return env


def output_for(job,config):
    paths=list((Path(config['work'])/'runs/wandb').glob(f"run-*-{job['wandb_id']}/files/training"))
    check(len(paths)<=1,'Multiple local outputs for a single scientific run')
    return paths[0] if paths else None


def validate_job(job):
    assigned=next((j for j in assignment()['jobs'] if j['name']==job.get('name')),None)
    check(assigned is not None and {k:v for k,v in job.items() if k!='gpu'}==assigned,
          'Changed assigned job/seed/flags in persistent state')


def validate_config(config,job):
    """Compare every parsed scientific argument, not just a few budget fields."""
    sys.path.insert(0,str(ROOT/'examples'))
    import battleship
    from _common import parse_args,validate_args
    parser=battleship.parser();expected=parse_args(parser,job['flags']);validate_args(parser,expected)
    administrative={'output_dir','wandb','wandb_project','wandb_run_name'}
    for key,value in vars(expected).items():
        if key not in administrative:
            check(config.get(key)==value,'Changed scientific configuration: '+key)


def run_logged(command,path,env):
    with path.open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        return process.wait()


def load_native_module(state,history):
    # Original native.py writes to its ROOT/reviewer_response. Override ONLY its
    # administrative output root: never overwrite hash-bound reference evidence.
    sys.path.insert(0,str(RUNS))
    spec=importlib.util.spec_from_file_location('sealed_history_native',RUNS/'native.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=state/'native'/f'h{history}'
    (module.ROOT/'reviewer_response').mkdir(parents=True,exist_ok=True)
    return module


def native(state,history):
    verify()
    module=load_native_module(state,history)
    sys.argv=[str(RUNS/'native.py'),'--history',str(history)]
    module.main()


def finalize(state,job):
    verify();validate_job(job);config=read(state/'settings.json');directory=state/'jobs'/job['name']
    check(read(directory/'training-exit.json',{}).get('exit_code')==0,
          'No successful training exit; finalization cannot train or resume a policy')
    output=output_for(job,config);check(output is not None,'W&B training output not found')
    cfg=read(output/'config.json')
    validate_config(cfg,job)
    rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    check([r['round'] for r in rows if 'round' in r]==list(range(1,51)),'Incomplete training history')
    pointers=list(output.glob('checkpoint*.archived.json'))
    check(len(pointers)==6 and all(read(p)['remote_committed_verified'] for p in pointers),
          'All original-cadence checkpoints must be archived')
    receipt={'status':'complete','job':{k:v for k,v in job.items() if k!='gpu'},
        'exit_code':0,'continuous_no_resume':True,'all_checkpoints_archived':True,
        'source_admission_sha256':assignment()['admission_sha256'],
        'lease_id':assignment()['lease_id'],'output':str(output),'finished_utc':utc(),
        'host':socket.gethostname(),'owner_host':config['owner_host'],'gpu':job['gpu'],
        'native_validation':read(state/'native'/f"h{job['history']}"/'reviewer_response'/f"native-h{job['history']}.json"),
        'python':sys.version,'runtime_versions':{p:importlib.metadata.version(p) for p in
            ('jax','jaxlib','jax-cuda12-plugin','jax-cuda12-pjrt','flax','optax','numpy','scipy','gymnax','wandb')},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'deployment_bundle_sha256':sha(HERE/'bundle-manifest.json')}
    bundle=directory/'result';bundle.mkdir(exist_ok=True)
    for file in [output/'config.json',output/'metrics.jsonl',output/'summary.json',*pointers]:
        destination=bundle/file.name
        if destination.exists():check(sha(destination)==sha(file),'Conflicting retained result evidence')
        else:shutil.copyfile(file,destination)
    # An explicit read-only finalize retry may refresh only administrative time.
    if (bundle/'receipt.json').exists():receipt=read(bundle/'receipt.json')
    else:write(bundle/'receipt.json',receipt)
    write(bundle/'index.json',{'lease_id':assignment()['lease_id'],'job':job['name'],
        'assignment_sha256':sha(HERE/'assignment.json'),
        'files':{p.name:sha(p) for p in bundle.iterdir() if p.name!='index.json'}})
    write(directory/'status.json',receipt)
    return bundle


def worker(state,name):
    verify();config=read(state/'settings.json');job=next(j for j in config['jobs'] if j['name']==name)
    validate_job(job)
    check(config['owner_host']==owner(),'State belongs to a different host')
    check(os.environ.get('CUDA_VISIBLE_DEVICES')==job['gpu']['uuid'],'Wrong GPU visibility')
    directory=state/'jobs'/name
    check(not (directory/'status.json').exists(),'Consumed worker attempt; never restart it')
    status={'status':'validating','job':job,'started_utc':utc(),'worker_pid':os.getpid()}
    write(directory/'status.json',status)
    try:
        check(job['gpu']['uuid'] not in occupied(),'Assigned GPU is busy')
        env=environment(job,config)
        code=run_logged([sys.executable,'-u','-B',str(HERE/'worker_queue.py'),'native','--state',str(state),
            '--history',str(job['history'])],directory/'native.console.txt',env)
        check(code==0,'Native validation failed; production was not launched')
        proof=read(state/'native'/f"h{job['history']}"/'reviewer_response'/f"native-h{job['history']}.json")
        check(proof['passed'] and proof['num_seeds']==5 and proof['native_updates']==64,'Invalid native proof')
        check(proof['cuda_visible_devices']==job['gpu']['uuid'],'Native proof used wrong GPU')
        # Short native evidence does NOT certify bitwise long-horizon continuity.
        check(job['gpu']['uuid'] not in occupied(),'GPU acquired by another process during validation')
        verify()
        import wandb
        a=assignment()
        check(not list(wandb.Api(timeout=120).runs(f"{a['entity']}/{a['project']}",
              filters={'name':job['wandb_id']},per_page=1)),
              'Scientific run ID already exists: preserve it, do not restart or replace it')
        with (directory/'production-started.json').open('x') as stream:
            json.dump({'utc':utc(),'no_resume':True,'wandb_id':job['wandb_id']},stream)
        status.update(status='running',native_passed=True);write(directory/'status.json',status)
        command=[sys.executable,'-u','-B','examples/battleship.py',*job['flags'],'--wandb',
            '--wandb-project',assignment()['project'],'--wandb-run-name',name]
        code=run_logged(command,directory/'training.console.txt',env)
        write(directory/'training-exit.json',{'exit_code':code,'utc':utc(),'command':command})
        check(code==0,'Training failed; preserve this attempt, never retry/resume it')
        finalize(state,job)
    except Exception as error:
        write(directory/'status.json',{**status,'status':'failed','error':repr(error),
            'traceback':traceback.format_exc(),'training_retry_allowed':False,'utc':utc()})
        raise


def cpu_env():
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONPATH=str(ROOT/'src'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('WANDB_SWEEP_ID',None)
    return env


def prepare_state(state,args):
    a=verify()
    check(not (RUNS/'lifecycle').exists(),'Do not run the B200 queue on the original training checkout')
    if (state/'settings.json').exists():
        config=read(state/'settings.json')
        check(config['owner_host']==owner() and config['assignment_sha256']==sha(HERE/'assignment.json'),
              'State/assignment belongs to another deployment')
        check(config['root']==str(ROOT),'Use the original checkout for this persistent deployment')
        check(len(config['jobs'])==8 and {j['name'] for j in config['jobs']}=={j['name'] for j in a['jobs']},
              'Changed persistent work allocation')
        for job in config['jobs']:validate_job(job)
        check(len({j['gpu']['uuid'] for j in config['jobs']})==8,'GPU assignments must remain distinct')
        return config
    selected=select_gpus(args.gpus,gpu_list())
    check(not ({g['uuid'] for g in selected}&occupied()),'A selected GPU is already occupied')
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                       if s.startswith('MemAvailable:')))*1024
    check(available>=96*2**30,'Need at least 96 GiB available host RAM for eight workers')
    work=Path(args.work or os.environ.get('HISTORY_B200_WORK') or state/'work').expanduser().resolve()
    work.mkdir(parents=True,exist_ok=True)
    check(shutil.disk_usage(work).free>=20*2**30,'Need at least 20 GiB free working storage')
    check('/dev/shm' not in str(work) and not str(work).startswith('/tmp/'),
          'Use durable home/scratch storage, not RAM or /tmp')
    for name in ('runs','cache','data','tmp'):(work/name).mkdir(exist_ok=True)
    jobs=[{**job,'gpu':gpu} for job,gpu in zip(a['jobs'],selected)]
    config={'owner_host':owner(),'owner_instance':secrets.token_hex(16),
        'hostname':socket.gethostname(),'created_utc':utc(),
        'assignment_sha256':sha(HERE/'assignment.json'),'work':str(work),'jobs':jobs,
        'root':str(ROOT),'python':sys.executable,'bundle_sha256':sha(HERE/'bundle-manifest.json')}
    write(state/'settings.json',config)
    return config


def preflight(state):
    verify();key=sha(HERE/'bundle-manifest.json')
    previous=read(state/'preflight.json',{})
    if previous.get('passed') and previous.get('bundle_sha256')==key:return
    path=state/'cpu-tests.console.txt'
    if path.exists():path=state/f'cpu-tests-{time.time_ns()}.console.txt'
    code=run_logged([sys.executable,'-B','-m','pytest','tests','reviewer_response/test_queue.py',
        'deployment/b200/tests','-q','--junitxml='+str(state/'cpu-tests.xml')],path,cpu_env())
    check(code==0,'CPU preflight failed: '+str(path))
    write(state/'preflight.json',{'passed':True,'bundle_sha256':key,'utc':utc(),
        'cpu_xml_sha256':sha(state/'cpu-tests.xml')})


def coordination(state):
    import wandb
    a=assignment();config=read(state/'settings.json')
    # Coordination can resume administrative logging only. Training IDs always
    # use resume=never and an exclusive local production-started marker.
    api=wandb.Api(timeout=120)
    existing=list(api.runs(f"{a['entity']}/{a['validation_project']}",
        filters={'name':a['coordination_run_id']},per_page=1))
    if existing:
        check(existing[0].config.get('owner_host')==config['owner_host'] and
              existing[0].config.get('owner_instance')==config['owner_instance'],
              'Assignment already claimed by another host/state; do not duplicate it')
    else:
        for job in a['jobs']:
            check(not list(api.runs(f"{a['entity']}/{a['project']}",filters={'name':job['wandb_id']},per_page=1)),
                  'A reserved scientific run already exists; inspect it rather than retry')
    run=wandb.init(entity=a['entity'],project=a['validation_project'],id=a['coordination_run_id'],
        resume='allow',name='B200 history deployment '+a['lease_id'],job_type='b200-history-coordination',
        config={'owner_host':config['owner_host'],'hostname':config['hostname'],
                'owner_instance':config['owner_instance'],
                'lease_id':a['lease_id'],'assignment_sha256':config['assignment_sha256'],
                'production_training':False,'training_resume_allowed':False})
    if not (state/'cloud-roundtrip.json').exists():
        artifact=wandb.Artifact('history-b200-deployment-'+a['lease_id'],type='source')
        for file in ('assignment.json','bundle-manifest.json','reservation-complete.json'):
            artifact.add_file(str(HERE/file),name=file,policy='immutable',skip_cache=True)
        uploaded=run.log_artifact(artifact);uploaded.wait()
        check(uploaded.state=='COMMITTED','Deployment artifact did not commit')
        restored=Path(api.artifact(uploaded.qualified_name).get_entry('assignment.json').download(
            root=str(state/'cloud-check'),skip_cache=True))
        check(sha(restored)==config['assignment_sha256'],'Cloud roundtrip hash mismatch')
        write(state/'cloud-roundtrip.json',{'passed':True,'artifact':uploaded.qualified_name,'utc':utc()})
    return run


def pane_states():
    r=subprocess.run(['tmux','list-panes','-a','-F','#{session_name}\t#{pane_dead}'],capture_output=True,text=True)
    return dict(line.split('\t') for line in r.stdout.splitlines())


def spawn(state,job,config):
    validate_job(job)
    directory=state/'jobs'/job['name'];directory.mkdir(parents=True,exist_ok=True)
    with (directory/'ticket.json').open('x') as stream:json.dump(job,stream,indent=2)
    check(job['session'] not in pane_states(),'tmux session already exists; do not hijack it')
    env=environment(job,config)
    subprocess.run(['tmux','new-session','-d','-s',job['session'],'-c',str(ROOT)],check=True)
    subprocess.run(['tmux','set-option','-t',job['session'],'remain-on-exit','on'],check=True)
    keys=('CUDA_VISIBLE_DEVICES','JAX_PLATFORMS','XLA_PYTHON_CLIENT_PREALLOCATE','PYTHONPATH',
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','WANDB_ENTITY','WANDB_MODE','WANDB_RUN_ID','WANDB_RESUME',
        'WANDB_RUN_GROUP','WANDB_JOB_TYPE','WANDB_TAGS','WANDB_NOTES','WANDB_DIR','WANDB_CACHE_DIR',
        'WANDB_DATA_DIR','TMPDIR','REBUTTAL_SOURCE_MANIFEST','REBUTTAL_CHECKPOINT_BYTES')
    command=shlex.join(['env','-u','WANDB_SWEEP_ID',*[f'{k}={env[k]}' for k in keys],
        sys.executable,'-u','-B',str(HERE/'worker_queue.py'),'worker','--state',str(state),'--job',job['name']])
    subprocess.run(['tmux','send-keys','-t',job['session'],'exec '+command,'Enter'],check=True)
    print('STARTED',job['session'],'GPU',job['gpu']['index'],'run',job['wandb_id'],flush=True)


def snapshot(state):
    config=read(state/'settings.json');panes=pane_states();items=[]
    for job in config['jobs']:
        directory=state/'jobs'/job['name'];record=read(directory/'status.json',{})
        phase=record.get('status','starting' if (directory/'ticket.json').exists() else 'queued')
        if phase in ('starting','validating','running') and panes.get(job['session'])!='0':phase='worker-lost'
        last={};output=output_for(job,config)
        if output and (output/'metrics.jsonl').exists():
            for line in (output/'metrics.jsonl').read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if 'round' in row:last=row
        items.append({'history':job['history'],'status':phase,'gpu':job['gpu']['index'],
            'session':job['session'],'round':last.get('round'),'wandb_id':job['wandb_id'],
            'result_published':(directory/'published.json').exists(),
            'error':record.get('error')})
    result={'utc':utc(),'state':str(state),'jobs':items,'complete':all(j['status']=='complete' and
            j['result_published'] for j in items),'training_retry_allowed':False}
    write(state/'status.json',result)
    return result


def publish_results(state,run):
    import wandb
    config=read(state/'settings.json')
    for job in config['jobs']:
        directory=state/'jobs'/job['name']
        if read(directory/'status.json',{}).get('status')!='complete' or (directory/'published.json').exists():continue
        bundle=directory/'result'
        artifact=wandb.Artifact('history-b200-result-'+job['wandb_id'],type='history-worker-results',
            metadata={'lease_id':assignment()['lease_id'],'training_run_id':job['wandb_id']})
        for path in sorted(bundle.iterdir()):artifact.add_file(str(path),name=path.name,policy='immutable',skip_cache=True)
        uploaded=run.log_artifact(artifact);uploaded.wait()
        check(uploaded.state=='COMMITTED','Results artifact did not commit')
        write(directory/'published.json',{'artifact':uploaded.qualified_name,'utc':utc()})
        print('PUBLISHED',job['name'],uploaded.qualified_name,flush=True)


def start(state,args):
    state.mkdir(parents=True,exist_ok=True)
    with (state/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare_state(state,args)
        check(config['bundle_sha256']==sha(HERE/'bundle-manifest.json'),'Do not update deployment during training')
        preflight(state)
        # Paths must be set BEFORE importing W&B, and inherited by its service.
        for key,value in environment(config['jobs'][0],config).items():
            if key in ('WANDB_DIR','WANDB_CACHE_DIR','WANDB_DATA_DIR','TMPDIR'):os.environ[key]=value
        os.environ.pop('WANDB_SWEEP_ID',None)
        run=coordination(state)
        try:
            for job in config['jobs']:
                directory=state/'jobs'/job['name']
                if not (directory/'ticket.json').exists():spawn(state,job,config)
            while True:
                verify();publish_results(state,run);status=snapshot(state)
                print(json.dumps(status),flush=True)
                if status['complete']:break
                # Failure is visible and retained, but healthy running learners
                # and result publication are never interrupted by the observer.
                time.sleep(30)
        finally:run.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','status','worker','native','finalize','install-check'))
    parser.add_argument('--state');parser.add_argument('--work');parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--job');parser.add_argument('--history',type=int);args=parser.parse_args()
    state=state_path(args.state)
    if args.action=='install-check':
        if (state/'settings.json').exists():
            for j in read(state/'settings.json')['jobs']:
                check(not (state/'jobs'/j['name']/'ticket.json').exists(),
                      'Do not reinstall/update dependencies after worker attempts have started')
        print('Installation permitted; no B200 worker attempts consumed');return
    if args.action=='status':
        check((state/'settings.json').exists(),'Queue has not been initialized at '+str(state))
        print(json.dumps(snapshot(state),indent=2));return
    if args.action=='start':start(state,args)
    elif args.action=='worker':worker(state,args.job)
    elif args.action=='native':native(state,args.history)
    else:
        job=next(j for j in read(state/'settings.json')['jobs'] if j['name']==args.job)
        print(finalize(state,job))


if __name__=='__main__':main()
