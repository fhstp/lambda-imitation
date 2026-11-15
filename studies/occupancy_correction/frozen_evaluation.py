"""Fixed-mu held-out scores, summed globally with gamma**real_episode_age.

No sampled-policy visitation is substituted for the fixed collector here.
Each fit is scored on the same immutable held-out dataset and frozen actor.
These are fitted neural disagreements/residual diagnostics, not exact Q errors.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from lambda_imitation.actor_critic import Head


def create_evaluator(settings, fns):
    actions = 25 if settings.environment == "battleship" else 36
    width = settings.hidden_dim
    actor_graph, _ = nnx.split(Head(width, (width,), actions, rngs=nnx.Rngs(0)))
    critic_graph, _ = nnx.split(Head(width, (width,), actions, layer_norm=True, rngs=nnx.Rngs(0)))

    def values(params, z):
        return jnp.stack([nnx.merge(critic_graph, p)(z) for p in params], axis=-1)

    @jax.jit
    def chunk(agent, info, lengths):
        z = fns.features(agent.feature_extractor, info, lengths)
        flat_z = z.reshape(-1, width)
        logits = nnx.merge(actor_graph, agent.actor)(flat_z)
        if settings.environment == "battleship":
            legal = info["observations"][..., 1:].reshape(-1, actions).astype(bool)
            legal |= ~legal.any(-1, keepdims=True)
            logits = jnp.where(legal, logits, -1e9)
        pi = jax.nn.softmax(logits).reshape((*z.shape[:2], actions))
        q1 = values(agent.lambda1_critic, flat_z).reshape((*z.shape[:2], actions, 2))
        q2 = values(agent.lambda2_critic, flat_z).reshape(q1.shape)
        delta = q1.min(-1)-q2.min(-1)
        d2 = (pi*delta**2).sum(-1)
        twin2 = (pi * 0.5*((q1[..., 0]-q1[..., 1])**2+(q2[..., 0]-q2[..., 1])**2)).sum(-1)
        valid = jnp.arange(z.shape[1])[None, :] < lengths[:, None]
        weights = valid*settings.gamma**jnp.arange(z.shape[1])[None, :]
        return jnp.array([(weights*d2).sum(), (weights*twin2).sum(), weights.sum(),
                          (valid*d2).sum(), valid.sum()])

    def evaluate(agent, heldout, batch_size=128):
        results = []
        for seed in range(heldout.lengths.shape[0]):
            single = jax.tree.map(lambda x: x[seed], agent)
            replay = jax.tree.map(lambda x: x[seed], heldout)
            lengths = np.asarray(replay.lengths)
            slots = np.flatnonzero(lengths)
            total = np.zeros(5, dtype=np.float64)
            for start in range(0, len(slots), batch_size):
                indices = slots[start:start+batch_size]
                chosen_lengths = np.zeros(batch_size, np.int32)
                chosen_lengths[:len(indices)] = lengths[indices]
                indices = np.pad(indices, (0, batch_size-len(indices)), mode="constant")
                data = {name: value[indices] for name, value in replay.info.items()}
                total += np.asarray(chunk(single, data, jnp.asarray(chosen_lengths)), np.float64)
            if not np.isfinite(total).all() or total[2] <= 0:
                raise FloatingPointError("invalid fixed-mu held-out evaluation; no seeds omitted")
            results.append({"discrepancy_rms": float(np.sqrt(total[0]/total[2])),
                            "auxiliary_twin_rms": float(np.sqrt(total[1]/total[2])),
                            "uniform_start_discrepancy_rms": float(np.sqrt(total[3]/total[4])),
                            "scored_transitions": int(total[4])})
        return results

    return evaluate
