"""Partial Minesweeper on the shared recurrent actor–critic experiment runner."""

from _common import Experiment, build_parser, parse_args, run


def build_experiment(args):
    import jax
    import jax.numpy as jnp
    from lambda_imitation.envs.minesweeper import MineSweeper, _count_neighbors
    from _probes import ProbeSpec

    if min(args.projection_dim, args.head_dim) < 1:
        raise ValueError("projection-dim and head-dim must be positive")
    env = MineSweeper(args.rows, args.cols, args.mines)

    def history_policy(obs, state, key):
        known_zero = state.viewed & (state.neighbor_counts == 0)
        safe = ~state.viewed & (_count_neighbors(known_zero) > 0)
        candidates = jnp.where(jnp.any(safe), safe, ~state.viewed).reshape(-1)
        return jax.random.categorical(key, jnp.where(candidates, 0.0, -1e9))

    probe = ProbeSpec(
        truth=lambda s: jax.nn.one_hot(s.neighbor_counts.reshape(-1), env.num_clues).reshape(-1),
        observed=lambda s: s.viewed.reshape(-1), shape=(args.rows, args.cols),
        title="Expected clue", classes=env.num_clues)
    return Experiment(env, args.projection_dim, (args.head_dim,), (args.head_dim,),
                      env.episode_length, reference=history_policy, probe=probe,
                      reset_memory_each_round=False)


def parser():
    p = build_parser("Minesweeper", dict(
        rounds=40, train_steps=5000, seed=1000100, memory_hidden_dim=128,
        batch_size=32, burn_in_length=30, sequence_length=32, lambda_truncation=32,
        fe_lr=1e-4, alpha=0.01, lambda1=0.0, lambda2=0.95, eval_episodes=128))
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--mines", type=int, default=6)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=128)
    return p


if __name__ == "__main__":
    p = parser()
    run(p, parse_args(p), build_experiment)
