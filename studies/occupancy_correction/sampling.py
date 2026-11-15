"""Sample actual replay anchors, not just change future trace coefficients.

For complete independent episodes from ONE fixed behaviour, the empirical
discounted starting measure gives episode e, time t mass gamma**t / Z, with
Z summed over ALL valid transitions. This converges to the nonterminal part
of the discounted occupation measure from the declared episode initial law.
Conditioning on an observation/action gives its empirical posterior. Finite
data are not asserted to equal the population posterior.

Uniform episode selection followed by separately normalized within-episode
time selection is NOT this measure when episode lengths differ. Nor is an
age-weighted mixture of adaptive behaviours certified as one stationary mu.

Only sampled anchors may be scored by a downstream loss. Returning a complete
episode preserves the raw prefix for recurrent reconstruction and the real
future through termination; it does not authorize scoring every returned row
as if each had been drawn from the anchor distribution.
"""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.tree_util.register_dataclass,
         data_fields=["info", "lengths"], meta_fields=["behaviour_policy_id"])
@dataclass(frozen=True)
class CompleteEpisodes:
    """Padded raw episodes and the identifier of their fixed collector.

    The identifier is a provenance assertion, not proof that the collector was
    fixed. Production must bind it to an immutable policy/configuration and
    record the collection law and initial/terminal conventions independently.
    """

    info: dict[str, jax.Array]
    lengths: jax.Array
    behaviour_policy_id: str


class AnchorBatch(NamedTuple):
    info: dict[str, jax.Array]
    lengths: jax.Array
    episode_indices: jax.Array
    anchor_times: jax.Array
    probabilities: jax.Array
    valid: jax.Array
    prefix_mask: jax.Array
    continuation_mask: jax.Array


def prepare_episodes(info, lengths, *, behaviour_policy_ids):
    """Validate complete episodes on the host, then place them on the device.

    Required scalar fields have shape (episodes, max_steps). Observations may
    have trailing dimensions. Zero-length slots are permitted but unsampleable.
    Each nonempty episode must end in exactly one real terminal. Incomplete
    in-flight episodes must stay outside this dataset; a learner budget boundary
    must not be relabeled as an environment terminal to pass validation.
    """
    required = {"observations", "actions", "rewards", "terminated", "behaviour_probs"}
    missing = required - set(info)
    if missing:
        raise ValueError(f"missing episode fields: {sorted(missing)}")
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("episode lengths must be a one-dimensional integer array")
    observations = np.asarray(info["observations"])
    if observations.ndim < 2 or observations.shape[0] != len(lengths):
        raise ValueError("observations must start with (episodes, max_steps)")
    shape = observations.shape[:2]
    if shape[1] < 1 or np.any(lengths < 0) or np.any(lengths > shape[1]):
        raise ValueError("episode lengths must fit the padded time dimension")
    if not np.any(lengths > 0):
        raise ValueError("no complete episode to sample")
    if len(behaviour_policy_ids) != len(lengths):
        raise ValueError("one behaviour policy identifier is required per slot")
    policies = {policy for policy, length in zip(behaviour_policy_ids, lengths) if length > 0}
    if len(policies) != 1 or any(not isinstance(p, str) or not p for p in policies):
        raise ValueError("theory-aligned sampling requires one identified fixed behaviour, not a mixture")

    valid = np.arange(shape[1])[None, :] < lengths[:, None]
    arrays = {}
    for name, value in info.items():
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[:2] != shape:
            raise ValueError(f"{name} must start with {shape}")
        if array.dtype.kind not in "bifu" or not np.isfinite(array[valid]).all():
            raise ValueError(f"{name} contains unsupported or nonfinite active data")
        if name in required - {"observations"} and array.ndim != 2:
            raise ValueError(f"{name} must be a scalar at each transition")
        arrays[name] = array
    terminals = arrays["terminated"]
    expected_terminals = valid & (np.arange(shape[1])[None, :] == lengths[:, None] - 1)
    if not np.array_equal(terminals[valid], expected_terminals[valid]):
        raise ValueError("every episode must have exactly one terminal at its actual end")
    probs = arrays["behaviour_probs"][valid]
    if np.any(probs <= 0) or np.any(probs > 1):
        raise ValueError("logged behaviour probabilities must be positive and at most one")
    actions = arrays["actions"][valid]
    if np.any(actions < 0) or not np.equal(actions, np.floor(actions)).all():
        raise ValueError("actions must be nonnegative discrete indices")

    sanitized = {}
    for name, array in arrays.items():
        mask = valid.reshape((*shape, *((1,) * (array.ndim - 2))))
        # Neutral padding must not create NaNs or leak another episode's data.
        padding = 1 if name in ("terminated", "behaviour_probs") else 0
        sanitized[name] = jnp.asarray(np.where(mask, array, padding))
    return CompleteEpisodes(sanitized, jnp.asarray(lengths, dtype=jnp.int32), policies.pop())


def anchor_probabilities(lengths, max_steps, *, gamma, measure):
    """Global probabilities over (episode, time), zero on every padded slot.

    ``uniform`` is uniform over valid transitions, NOT historical uniform
    contiguous-window replay. It is the matched complete-episode control for
    this sampler. It must not be mislabeled an exact legacy reproduction.
    """
    if not np.isfinite(gamma) or not 0 < gamma < 1:
        raise ValueError("require a finite discount strictly between zero and one")
    if measure not in ("uniform", "discounted"):
        raise ValueError("measure must be uniform or discounted")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    ages = jnp.arange(max_steps)
    valid = ages[None, :] < lengths[:, None]
    time_weights = gamma ** ages if measure == "discounted" else jnp.ones(max_steps)
    weights = jnp.where(valid, time_weights[None, :], 0.0)
    # Admission rejects empty datasets; no fallback to a fictitious uniform law.
    return weights / weights.sum()


def create_anchor_sampler(*, batch_size, gamma, measure):
    """Return a JIT-compatible sampler that really changes replay selection.

    Draw with replacement from the global anchor law. Return whole raw
    episodes, anchor times, and proposal probabilities. The consumer must
    reconstruct from episode start and score only the selected anchors.
    ``mu_logged`` is returned untouched on active positions for the separate
    future-trace correction. No action importance ratio is clipped here.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    # Validate static arguments before entering JIT.
    anchor_probabilities(jnp.array([1]), 1, gamma=gamma, measure=measure)

    @jax.jit
    def sample(episodes, key):
        max_steps = episodes.info["observations"].shape[1]
        probabilities = anchor_probabilities(
            episodes.lengths, max_steps, gamma=gamma, measure=measure)
        flat = jax.random.choice(key, probabilities.size, shape=(batch_size,),
                                 replace=True, p=probabilities.reshape(-1))
        indices, times = flat // max_steps, flat % max_steps
        lengths = episodes.lengths[indices]
        ages = jnp.arange(max_steps)[None, :]
        valid = ages < lengths[:, None]
        return AnchorBatch(
            {name: value[indices] for name, value in episodes.info.items()},
            lengths, indices, times, probabilities[indices, times], valid,
            valid & (ages <= times[:, None]), valid & (ages >= times[:, None]))

    return sample


def at_anchors(time_values, batch):
    """Select precisely the sampled positions from batch-major (B,T,...) data."""
    return time_values[jnp.arange(batch.anchor_times.shape[0]), batch.anchor_times]
