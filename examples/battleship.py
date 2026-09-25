"""5×5 Battleship experiment. See examples/README.md for paper configurations."""

from _common import Experiment, build_parser, parse_args, run


def make_bayes_policy(rows, cols, ship_lengths, hit_weight=12.0):
    """Greedy placement-density reference using only the observed shot record.

    Each non-miss-overlapping placement receives weight ``hit_weight ** hits``.
    Ties break by row-major index. This is the paper's reference heuristic,
    not an exact joint posterior or an optimal policy.
    """
    import jax.numpy as jnp
    import numpy as np

    grid = np.arange(rows * cols).reshape(rows, cols)
    placements = []
    for length in sorted(set(ship_lengths)):
        cells = [grid[r, c:c + length] for r in range(rows) for c in range(cols - length + 1)]
        cells += [grid[r:r + length, c] for c in range(cols) for r in range(rows - length + 1)]
        placements.append(jnp.asarray(np.stack(cells)))

    def policy(obs, state, key):
        hm = state.hits_misses.reshape(-1)
        density = jnp.zeros(rows * cols)
        for cells in placements:
            shots = hm[cells]
            weight = hit_weight ** jnp.sum(shots == 2, -1) * ~jnp.any(shots == 1, -1)
            density = density.at[cells.reshape(-1)].add(jnp.repeat(weight, cells.shape[1]))
        return jnp.argmax(jnp.where(hm == 0, density, -1.0))

    return policy


def build_experiment(args):
    from lambda_imitation.envs.battleship import Battleship
    from lambda_imitation.utils import battleship_projection
    from _probes import ProbeSpec

    lengths = tuple(int(x) for x in args.ship_lengths.split(","))
    env = Battleship(args.rows, args.cols, lengths)
    probe = ProbeSpec(
        truth=lambda s: s.board.reshape(-1).astype(float),
        observed=lambda s: (s.hits_misses > 0).reshape(-1),
        shape=(args.rows, args.cols), title="P(ship)")
    return Experiment(
        env, battleship_projection(args.memory_hidden_dim, extra_layer=args.memory_type == "identity"),
        (args.memory_hidden_dim,), (args.memory_hidden_dim,), args.rows * args.cols + 1,
        obs_fn=lambda o: o[..., :1], mask_fn=lambda o: o[..., 1:],
        reference=make_bayes_policy(args.rows, args.cols, lengths), probe=probe)


def parser():
    p = build_parser("Battleship")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--ship-lengths", default="3,2", help="comma-separated ship lengths")
    return p


if __name__ == "__main__":
    p = parser()
    run(p, parse_args(p), build_experiment)
