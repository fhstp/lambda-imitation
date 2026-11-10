"""Full production dimensions, short native training and read-only evaluation.

Validation state is never loaded by production. Native values are not results.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import battleship
from _common import validate_args,make_evaluator
from protocol import flags


def main():
    p=argparse.ArgumentParser();p.add_argument('--history',type=int,required=True);args=p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lambda_imitation import Hyperparameters,create_actor_critic_from_env,env_spec_from_gymnax
    from _common import parse_args
    parser=battleship.parser()
    config=parse_args(parser,flags('battleship','retrace',7451234536,5)+[
        '--history-length',str(args.history),'--history-context','26'])
    validate_args(parser,config);experiment=battleship.build_experiment(config)
    hp=Hyperparameters(**{k:getattr(config,k) for k in Hyperparameters._fields if k!='lambda_critic_lr'},
                       lambda_critic_lr=config.critic_lr)
    env=experiment.env;params=env.default_params;spec=env_spec_from_gymnax(env,params)
    def build(seed):
        return create_actor_critic_from_env(spec,hp=hp,projection=experiment.projection,
            memory_type='gru',memory_hidden_dim=512,use_prev_action=True,
            actor_dims=(512,),critic_dims=(512,),lambda1_critic_dims=(512,),lambda2_critic_dims=(512,),
            train_steps=32,approximate_lambda=True,critic_layer_norm=True,
            obs_fn=experiment.obs_fn,mask_fn=experiment.mask_fn,seed=seed,history_length=args.history)
    states=[];started=time.monotonic()
    for seed in range(7451234536,7451234541):
        state,fns=build(seed);states.append(state)
    state=jax.tree.map(lambda *x:jnp.stack(x),*states)
    initial_hash=hashlib.sha256()
    for x in jax.tree.leaves(state[:5]):initial_hash.update(np.asarray(x).tobytes())
    keys=jnp.stack([jax.random.PRNGKey(s) for s in range(7451234536,7451234541)])
    split=jax.vmap(lambda k:jax.random.split(k,3))(keys)
    _,es=jax.vmap(lambda k:env.reset(k,params))(split[:,1])
    prefill=jax.jit(jax.vmap(lambda s,e,k:fns.prefill_buffer(s,env,params,e,12800,k)))
    state,_=prefill(state,es,split[:,2])
    prefill_hash=hashlib.sha256()
    for x in jax.tree.leaves(state.online_buffer):prefill_hash.update(np.asarray(x).tobytes())
    _,es=jax.vmap(lambda k:env.reset(k,params))(split[:,1])
    carry=jnp.zeros((5,args.history*27));pa=jnp.zeros((5,25))
    train=jax.jit(jax.vmap(lambda s,e,c,a,k:fns.train_unrolled(s,env,params,e,c,a,k)))
    start=time.monotonic();out=train(state,es,carry,pa,split[:,0]);jax.block_until_ready(out)
    first=time.monotonic()-start
    start=time.monotonic();out2=train(*out[:4],split[:,0]);jax.block_until_ready(out2)
    second=time.monotonic()-start
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((out2[0][:15],out2[-1])))
    assert np.all(np.asarray(out2[0].update_step)==64)
    evaluation={}
    for mode in ('greedy','sampled'):
        evaluator=make_evaluator(experiment,fns,args.history*27,25,50,mode)
        values=evaluator(out2[0],keys)
        for metric in ('return','steps','completed','discrepancy_rms','aux_twin_rms','repeat_fraction'):
            assert np.isfinite(np.asarray(values[metric])).all()
        assert np.all(np.asarray(values['completed'])==1)
        assert np.all(np.asarray(values['repeat_fraction'])==0)
        evaluation[mode]={k:np.asarray(v).tolist() for k,v in values.items() if k!='known_safe_choice'}
    result={'passed':True,'history':args.history,'num_seeds':5,'batch_size':128,'learning_steps':40,
        'context_only_prefix':26,'burn_in':10,'tail':50,'prefill':12800,'native_updates':64,
        'devices':[str(d) for d in jax.devices()],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'initial_network_sha256':initial_hash.hexdigest(),'prefill_sha256':prefill_hash.hexdigest(),
        'first32_updates_including_compilation_seconds':first,'second32_updates_seconds':second,
        'steady_seconds_per_update':second/32,'evaluation':evaluation,'validation_only_no_state_reuse':True,
        'total_seconds':time.monotonic()-started}
    (ROOT/'reviewer_response'/f'native-h{args.history}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
