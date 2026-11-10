"""Exact rolling-window GRU: H previous network inputs plus the current input.

Only raw outcome/action inputs are retained, never a learned state from before
the window. Invalid left padding does not advance the GRU. The paper's action
mask remains external and therefore is NOT subject to this representation cap.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .utils import RecurrentFeatureExtractor


class HistoryFeatureExtractor(nnx.Module):
    def __init__(self, input_shape, projection, hidden_dim, prev_action_dim, history_length, *, rngs):
        if history_length < 0:
            raise ValueError('history_length must be nonnegative')
        self.history_length=history_length
        self.input_shape=tuple(input_shape)
        self.obs_dim=1
        for d in input_shape:self.obs_dim*=d
        self.prev_action_dim=prev_action_dim
        self.token_dim=self.obs_dim+prev_action_dim+1
        self.core=RecurrentFeatureExtractor(input_shape,projection,'gru',hidden_dim,
                                           prev_action_dim,rngs=rngs)
        self.output_dim=hidden_dim
        self.carry_dim=history_length*self.token_dim

    def initialize_carry(self,batch_size):
        return jnp.zeros((batch_size,self.carry_dim),jnp.float32)

    def encode_windows(self,tokens):
        """tokens: (batch,H+1,obs+previous-action+validity)."""
        def step(carry,token):
            obs=token[:,:self.obs_dim].reshape((-1,*self.input_shape))
            pa=token[:,self.obs_dim:-1]
            new,y=self.core(carry,obs,pa)
            carry=jnp.where(token[:,-1:]>0,new,carry)
            return carry,None
        initial=jnp.zeros((tokens.shape[0],self.output_dim),tokens.dtype)
        carry,_=jax.lax.scan(jax.checkpoint(step),initial,jnp.swapaxes(tokens,0,1))
        return carry

    def __call__(self,carry,obs,prev_action=None):
        if prev_action is None:
            prev_action=jnp.zeros((obs.shape[0],self.prev_action_dim),obs.dtype)
        current=jnp.concatenate((obs.reshape((obs.shape[0],-1)),prev_action,
                                 jnp.ones((obs.shape[0],1),obs.dtype)),-1)
        history=carry.reshape((obs.shape[0],self.history_length,self.token_dim))
        window=jnp.concatenate((history,current[:,None,:]),axis=1)
        y=self.encode_windows(window)
        next_carry=window[:,1:,:].reshape((obs.shape[0],self.carry_dim))
        return next_carry,y

    def sequence(self,observations,actions,dones,start=0):
        """Time-major raw inputs, exact independent windows, episode-safe.

        actions are one-hot actions AT t. Network input at t uses action t-1,
        zeroed at episode starts. Replay supplies a common context-only prefix.
        """
        obs=observations.reshape((*observations.shape[:2],-1))
        reset=jnp.concatenate((jnp.ones_like(dones[:1]),dones[:-1]),axis=0).astype(bool)
        pa=jnp.concatenate((jnp.zeros_like(actions[:1]),actions[:-1]),axis=0)
        pa=jnp.where(reset[...,None],0,pa)
        tokens=jnp.concatenate((obs,pa,jnp.ones((*obs.shape[:2],1),obs.dtype)),axis=-1)
        segment=jnp.cumsum(reset,axis=0)
        t=jnp.arange(start,len(obs))[:,None]
        indices=t+jnp.arange(-self.history_length,1)[None,:]
        clipped=jnp.maximum(indices,0)
        windows=tokens[clipped]
        valid=(indices>=0)[...,None] & (segment[clipped]==segment[start:,None,:])
        windows=windows.at[...,-1].set(valid)
        windows=jnp.swapaxes(windows,1,2)
        features=self.encode_windows(windows.reshape((-1,self.history_length+1,self.token_dim)))
        return features.reshape((len(obs)-start,obs.shape[1],self.output_dim))
