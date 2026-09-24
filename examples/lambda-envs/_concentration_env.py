"""Independent gymnax/JAX version of POPGym's rank-matching Concentration.

Rules reference: https://github.com/proroklab/popgym/blob/master/popgym/envs/concentration.py
No POPGym dependency is needed. A balanced, shuffled deck has ``num_types`` ranks
with an even number of cards of each rank. One action flips one position. A
successful pair earns ``1 / (num_cards // 2)``; a failed attempt costs
``1 / (2 * num_cards)`` per attempted flip, including duplicate selections.

The default observation is the flattened, float32 one-hot encoding of the
visible board: ``num_types`` rank channels followed by one hidden-card channel
at each position. Like POPGym's step observation, it shows both attempted cards
on a failed match until the NEXT action. The pre-resolution labels are cached
in the state so ``get_obs(state)`` agrees with the observation returned by step.

``remember=True`` is an explicit history-table control, not hidden-state access.
Its flat observation concatenates [default observation, one-hot seen-rank table,
permanently-matched mask, current-first-card mask]. Unseen table entries use the
hidden channel. The masks describe the POST-resolution phase, so a memoryless
policy can distinguish completed attempts from a pending first selection.
Widths are ``N * (K + 1)`` normally and ``2 * N * (K + 1) + 2 * N`` with memory
(728 and 1560 for the default 52 cards / 13 ranks). No action mask is applied.

``Environment.step`` supplies gymnax's usual terminal auto-reset; ``step_env``
returns the terminal board itself. ``diagnostics`` describes the supplied state
and proposed action, so call it before stepping when counting opportunities.
"""

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class ConcentrationParams(NamedTuple):
    max_steps_in_episode: int = 104


class ConcentrationState(NamedTuple):
    """Immutable pytree; all vectors have length ``num_cards``.

    ``cards`` contains private int32 rank labels in [0, num_types).
    ``face_up`` and ``in_play`` are boolean permanent-match / first-card masks.
    ``timestep`` is a scalar int32 count of actions taken. ``visible`` caches
    the last emitted board as int32 labels, with ``num_types`` meaning hidden.
    ``seen`` records whether a position has ever been revealed; ``last_seen``
    stores its revealed rank (NOT a timestamp), or ``num_types`` if unseen.
    Neither ``get_obs`` nor ``diagnostics`` reads the private ``cards`` array.
    """

    cards: jax.Array
    face_up: jax.Array
    in_play: jax.Array
    timestep: jax.Array
    visible: jax.Array
    seen: jax.Array
    last_seen: jax.Array


class Concentration(environment.Environment):
    """Rank-matching Concentration with balanced, even rank multiplicities.

    Smaller decks, e.g. ``Concentration(num_cards=8, num_types=2)``, use the
    same rules and an episode limit of ``2 * num_cards`` actions. ``num_cards``
    must be divisible by ``num_types`` with a positive even quotient. Actions
    are scalar position indices in [0, num_cards), including already-matched
    positions and repeated first-card selections (both incur penalties).
    """

    obs_requires_prev_action = True

    def __init__(self, num_cards=52, num_types=13, remember=False):
        super().__init__()
        try:
            self.num_cards = operator.index(num_cards)
            self.num_types = operator.index(num_types)
        except TypeError as exc:
            raise ValueError("num_cards and num_types must be integers") from exc
        if self.num_cards < 2 or self.num_cards % 2:
            raise ValueError("num_cards must be a positive even integer")
        if self.num_types < 1 or self.num_cards % self.num_types:
            raise ValueError("num_types must be positive and divide num_cards")
        if (self.num_cards // self.num_types) % 2:
            raise ValueError("Each rank must have an even multiplicity")
        self.remember = bool(remember)
        self.episode_length = 2 * self.num_cards
        self.success_reward_scale = 1.0 / (self.num_cards // 2)
        self.failure_reward_scale = -1.0 / self.episode_length
        width = self.num_cards * (self.num_types + 1)
        self.obs_shape = (2 * width + 2 * self.num_cards if self.remember else width,)

    @property
    def default_params(self) -> ConcentrationParams:
        return ConcentrationParams(max_steps_in_episode=self.episode_length)

    @property
    def num_actions(self) -> int:
        return self.num_cards

    def reset_env(self, key, params):
        cards = jax.random.permutation(
            key, jnp.arange(self.num_cards, dtype=jnp.int32) % self.num_types)
        empty = jnp.zeros(self.num_cards, dtype=jnp.bool_)
        hidden = jnp.full(self.num_cards, self.num_types, dtype=jnp.int32)
        state = ConcentrationState(
            cards=cards, face_up=empty, in_play=empty,
            timestep=jnp.int32(0), visible=hidden, seen=empty, last_seen=hidden)
        return self.get_obs(state, params), state

    def step_env(self, key, state, action, params):
        action = jnp.asarray(action, dtype=jnp.int32)
        has_first = jnp.any(state.in_play)
        first = jnp.argmax(state.in_play)
        attempted = state.in_play.at[action].set(True)

        # Cache the reveal BEFORE resolving the attempt. Old failed pairs are
        # absent from in_play, so they disappear when the next card is flipped.
        visible_mask = state.face_up | attempted
        visible = jnp.where(visible_mask, state.cards, self.num_types)
        seen = state.seen | visible_mask
        last_seen = jnp.where(visible_mask, visible, state.last_seen)

        invalid = state.face_up[action] | state.in_play[action]
        match = has_first & ~invalid & (state.cards[first] == state.cards[action])
        face_up = state.face_up | (match & attempted)
        in_play = attempted & ~has_first & ~invalid

        # Count attempts, not distinct positions: selecting the first card
        # again costs TWO flips even though attempted has only one True entry.
        num_attempts = 1 + has_first.astype(jnp.int32)
        reward = jnp.where(
            match, self.success_reward_scale,
            jnp.where(has_first | invalid,
                      num_attempts * self.failure_reward_scale, 0.0),
        ).astype(jnp.float32)
        state = state._replace(
            face_up=face_up, in_play=in_play, timestep=state.timestep + 1,
            visible=visible, seen=seen, last_seen=last_seen)
        done = self.is_terminal(state, params)
        return (self.get_obs(state, params), state, reward, done,
                {"discount": self.discount(state, params)})

    def get_obs(self, state, params=None):
        visible = jax.nn.one_hot(
            state.visible, self.num_types + 1, dtype=jnp.float32).reshape(-1)
        if not self.remember:
            return visible
        history = jax.nn.one_hot(
            state.last_seen, self.num_types + 1, dtype=jnp.float32).reshape(-1)
        return jnp.concatenate((visible, history,
                                state.face_up.astype(jnp.float32),
                                state.in_play.astype(jnp.float32)))

    def is_terminal(self, state, params=None):
        if params is None:
            params = self.default_params
        return jnp.all(state.face_up) | (state.timestep >= params.max_steps_in_episode)

    def diagnostics(self, state, action) -> dict[str, jax.Array]:
        """Pre-action history-only metrics (bool, bool, bool, int32 scalars).

        An opportunity exists only while a first card is pending and another
        previously revealed, unmatched card has the same recorded rank. Taking
        an unseen matching card does not count as a known match. ``pairs_matched``
        is cumulative in the supplied state, before the proposed action.
        """
        action = jnp.asarray(action, dtype=jnp.int32)
        first = jnp.argmax(state.in_play)
        known_matches = (state.seen & ~state.face_up & ~state.in_play
                         & (state.last_seen == state.last_seen[first]))
        opportunity = jnp.any(state.in_play) & state.seen[first] & jnp.any(known_matches)
        return {
            "known_match_opportunity": opportunity,
            "known_match_taken": opportunity & known_matches[action],
            "invalid_action": state.face_up[action] | state.in_play[action],
            "pairs_matched": jnp.sum(state.face_up, dtype=jnp.int32) // 2,
        }

    def action_space(self, params=None):
        return spaces.Discrete(self.num_cards)

    def observation_space(self, params=None):
        return spaces.Box(0.0, 1.0, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params=None):
        if params is None:
            params = self.default_params
        shape = (self.num_cards,)
        return spaces.Dict({
            "cards": spaces.Box(0, self.num_types - 1, shape, jnp.int32),
            "face_up": spaces.Box(0, 1, shape, jnp.bool_),
            "in_play": spaces.Box(0, 1, shape, jnp.bool_),
            "timestep": spaces.Discrete(params.max_steps_in_episode + 1),
            "visible": spaces.Box(0, self.num_types, shape, jnp.int32),
            "seen": spaces.Box(0, 1, shape, jnp.bool_),
            "last_seen": spaces.Box(0, self.num_types, shape, jnp.int32),
        })
