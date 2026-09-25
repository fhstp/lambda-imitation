"""PocMan experiment with an optional remaining-pellet memory probe."""

from _common import Experiment, build_parser, parse_args, run


def build_experiment(args):
    import jax
    import numpy as np
    from lambda_imitation.envs.pocman import PocMan, SMALLER_GAME_MAP
    from lambda_imitation.utils import relu_projection
    from _probes import ProbeSpec

    env = PocMan()
    _, state = env.reset(jax.random.key(0), env.default_params)
    cells = np.asarray(state.pellet_locations)[:, ::-1]
    walls = np.array([[c == "X" for c in row] for row in SMALLER_GAME_MAP])
    remaining = lambda s: (s.pellet_locations != 0).any(-1)
    probe = ProbeSpec(truth=lambda s: remaining(s).astype(float),
                      observed=lambda s: ~remaining(s), shape=walls.shape,
                      title="P(pellet)", cells=cells, walls=walls,
                      split_observed=False)
    return Experiment(env, relu_projection(args.memory_hidden_dim),
                      (args.memory_hidden_dim,), (args.memory_hidden_dim,),
                      env.time_limit, probe=probe)


def parser():
    return build_parser("PocMan", dict(
        gamma=0.95, critic_layer_norm=False, batch_size=512,
        sequence_length=20, burn_in_length=32, lambda_truncation=30,
        online_buffer_size=200_000, lambda1=0.1, lambda2=0.95,
        eval_episodes=10))


if __name__ == "__main__":
    p = parser()
    run(p, parse_args(p), build_experiment)
