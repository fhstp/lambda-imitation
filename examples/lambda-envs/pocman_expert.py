"""Scripted (privileged) PocMan expert.

The maze is static (``AsciiGenerator(SMALLER_GAME_MAP)``), so all-pairs shortest
paths are precomputed once on the host and the policy becomes a table lookup:

* walk toward the nearest remaining pellet (shortest-path distance, not Manhattan),
* never step within ``safety_margin`` steps of a ghost (flee if cornered),
* while a power pill is active, chase the nearest ghost (+200 per ghost eaten).

It reads the *full* env state -- privileged exactly like the Battleship Bayes
player -- while the learner still only ever sees the 11-dim observation.

The policy is jax-traceable, so it plugs straight into
``fns.prefill_buffer(..., behaviour_fn=policy)``: ``(obs, env_state, key) ->
(action_index, b(a|s))``.

Coordinate conventions (see the ``PocMan`` docstring):
    player_locations.x == row, player_locations.y == col
    pellet/power-up/ghost locations are (col, row)
Actions (verified against ``jumanji`` ``player_step``):
    0 = up (row-1), 1 = left (col-1), 2 = down (row+1), 3 = right (col+1)
"""

import argparse
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

# (d_row, d_col) per action index.
MOVES = ((-1, 0), (0, -1), (1, 0), (0, 1))
BIG = 9999  # stand-in for "unreachable" in the distance table


def build_distance_table(grid: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path distances over walkable cells (BFS per source).

    Cells are flat-indexed ``row * W + col``; walls and unreachable pairs get
    ``BIG``. Moves wrap around the maze edges (the row-9 tunnel), matching the
    env's ``% x_size`` / ``% y_size`` player step.
    """
    h, w = grid.shape
    n = h * w
    walkable = grid == 1
    neighbours = [[] for _ in range(n)]
    for r in range(h):
        for c in range(w):
            if not walkable[r, c]:
                continue
            for dr, dc in MOVES:
                rr, cc = (r + dr) % h, (c + dc) % w
                if walkable[rr, cc]:
                    neighbours[r * w + c].append(rr * w + cc)

    dist = np.full((n, n), BIG, dtype=np.int16)
    for src in range(n):
        if not walkable[src // w, src % w]:
            continue
        dist[src, src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            d = dist[src, cur] + 1
            for nxt in neighbours[cur]:
                if dist[src, nxt] > d:
                    dist[src, nxt] = d
                    queue.append(nxt)
    return dist


def make_pocman_expert(env, *, safety_margin: int = 5, chase: bool = True,
                       chase_min_time: int = 3, epsilon: float = 0.0,
                       rollout_depth: int = 28, death_penalty: float = 500.0,
                       gamma: float = 0.95):
    """Build the scripted policy. Returns ``fn(obs, state, key) -> (action, prob)``.

    With ``rollout_depth > 0`` the greedy table policy is used as the base of one
    step of policy improvement: each of the four first actions is simulated in the
    real env for ``rollout_depth`` steps of greedy continuation, and the action
    with the best discounted return (minus ``death_penalty`` if it dies) wins.
    Costs ``4 * rollout_depth`` env steps per decision.
    """
    _, state = jax.jit(env.reset)(jax.random.key(0), env.default_params)
    grid = np.array(state.grid)
    h, w = grid.shape
    dist = jnp.asarray(build_distance_table(grid).astype(np.int32))
    walkable = jnp.asarray(grid == 1)
    d_rows = jnp.asarray([m[0] for m in MOVES])
    d_cols = jnp.asarray([m[1] for m in MOVES])

    def greedy_scores(state):
        row, col = state.player_locations.x, state.player_locations.y
        cur = row * w + col

        nb_rows = (row + d_rows) % h
        nb_cols = (col + d_cols) % w
        nb = nb_rows * w + nb_cols
        valid = walkable[nb_rows, nb_cols]

        pellets = state.pellet_locations                      # (N, 2) as (col, row)
        pellet_flat = pellets[:, 1] * w + pellets[:, 0]
        alive = pellets.sum(axis=-1) > 0                      # eaten pellets are zeroed
        d_pellets = jnp.where(alive, dist[cur, pellet_flat], BIG)
        target = pellet_flat[jnp.argmin(d_pellets)]

        ghosts = state.ghost_locations                        # (4, 2) as (col, row)
        ghost_flat = ghosts[:, 1] * w + ghosts[:, 0]
        d_ghosts = dist[cur, ghost_flat]
        nearest_ghost = ghost_flat[jnp.argmin(d_ghosts)]

        frightened = state.frightened_state_time
        hunting = chase & (frightened > chase_min_time)

        goal = jnp.where(hunting, nearest_ghost, target)
        to_goal = -dist[nb, goal].astype(jnp.float32)
        ghost_gap = jnp.min(dist[nb][:, ghost_flat], axis=-1).astype(jnp.float32)

        # Never step within `safety_margin` of a live ghost; among the remaining
        # moves head straight for the goal. Cornered (no safe move) -> maximise
        # the ghost distance and break ties toward the goal.
        safe = valid & ((frightened > 0) | (ghost_gap >= safety_margin))
        scores = jnp.where(safe, to_goal, -jnp.inf)
        flee = jnp.where(valid, 100.0 * ghost_gap + to_goal, -jnp.inf)
        scores = jnp.where(jnp.any(safe), scores, flee)
        return scores, valid

    params = env.default_params
    sim_key = jax.random.key(0)  # unused by the env's ghost dynamics (they use state.key)

    def rollout_scores(state):
        """Discounted return of `first_action` followed by greedy play."""

        def one_branch(first_action):
            def step(carry, t):
                st, done, ret, disc = carry
                greedy = jnp.argmax(greedy_scores(st)[0])
                action = jnp.where(t == 0, first_action, greedy)
                _, st, reward, d, _ = env.step_env(sim_key, st, action, params)
                ret = ret + disc * reward * (1.0 - done)
                dead = jnp.maximum(done, (st.dead == 1).astype(jnp.float32))
                return (st, jnp.maximum(done, d.astype(jnp.float32)), ret,
                        disc * gamma), dead

            (st, _, ret, _), deaths = jax.lax.scan(
                step, (state, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0)),
                jnp.arange(rollout_depth))
            return ret - death_penalty * deaths[-1]

        return jax.vmap(one_branch)(jnp.arange(4))

    def policy(obs, state, key):
        del obs  # privileged: the expert reads the state, not the observation
        scores, valid = greedy_scores(state)
        if rollout_depth > 0:
            scores = jnp.where(valid, rollout_scores(state), -jnp.inf)

        greedy = jnp.argmax(scores)
        num_valid = jnp.maximum(valid.sum(), 1)
        explore_key, pick_key = jax.random.split(key)
        random_action = jax.random.choice(pick_key, 4, p=valid / num_valid)
        explore = jax.random.uniform(explore_key) < epsilon
        action = jnp.where(explore, random_action, greedy)

        prob = (1.0 - epsilon) * (action == greedy) + epsilon * valid[action] / num_valid
        return action.astype(jnp.int32), prob.astype(jnp.float32)

    return policy


def make_random_policy(env):
    """Uniform over the four actions (the env treats a wall move as a no-op)."""
    del env

    def policy(obs, state, key):
        del obs, state
        return jax.random.randint(key, (), 0, 4).astype(jnp.int32), jnp.float32(0.25)

    return policy


def evaluate(env, policy, num_episodes: int, seed: int = 0, gamma: float = 0.95):
    """Run ``num_episodes`` full episodes in parallel; no auto-reset."""
    params = env.default_params

    def one_episode(key):
        reset_key, key = jax.random.split(key)
        obs, state = env.reset(reset_key, params)

        def cond(carry):
            _, _, done, *_ = carry
            return jnp.logical_not(done)

        def body(carry):
            obs, state, _, ret, disc_ret, steps, key = carry
            act_key, step_key, key = jax.random.split(key, 3)
            action, _ = policy(obs, state, act_key)
            obs, state, reward, done, _ = env.step_env(step_key, state, action, params)
            return (obs, state, done, ret + reward, disc_ret + gamma ** steps * reward,
                    steps + 1, key)

        carry = (obs, state, jnp.bool_(False), jnp.float32(0.0), jnp.float32(0.0),
                 jnp.int32(0), key)
        obs, state, _, ret, disc_ret, steps, _ = jax.lax.while_loop(cond, body, carry)
        return {
            "return": ret,
            "discounted_return": disc_ret,
            "steps": steps.astype(jnp.float32),
            "pellets_eaten": (191 - state.pellets).astype(jnp.float32),
            "died": (state.dead == 1).astype(jnp.float32),
            "cleared": (state.pellets == 0).astype(jnp.float32),
        }

    keys = jax.random.split(jax.random.key(seed), num_episodes)
    out = jax.jit(jax.vmap(one_episode))(keys)
    return {k: (float(v.mean()), float(v.std())) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--safety-margin", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--rollout-depth", type=int, default=28,
                   help="0 = greedy table policy; >0 = rollout policy improvement")
    p.add_argument("--no-chase", action="store_true", help="ignore edible ghosts")
    args = p.parse_args()

    from lambda_envs.envs.pocman import PocMan

    env = PocMan()
    expert = make_pocman_expert(env, safety_margin=args.safety_margin,
                                chase=not args.no_chase, epsilon=args.epsilon,
                                rollout_depth=args.rollout_depth)
    random_policy = make_random_policy(env)

    rows = []
    for name, policy in (("random", random_policy), ("expert", expert)):
        stats = evaluate(env, policy, args.episodes, seed=args.seed)
        rows.append((name, stats))
        print(f"{name:>8}: " + "  ".join(
            f"{k}={m:.2f}+-{s:.2f}" for k, (m, s) in stats.items()))

    assert rows[1][1]["return"][0] > rows[0][1]["return"][0], "expert lost to random"


if __name__ == "__main__":
    main()
