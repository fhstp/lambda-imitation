# Bundled experiment environments

These modules ship in the package and use the Gymnax interface:

```python
import jax
from lambda_imitation.envs.battleship import Battleship

env = Battleship(rows=5, cols=5, ship_lengths=(3, 2))
obs, state = env.reset(jax.random.key(0), env.default_params)
obs, state, reward, done, info = env.step(
    jax.random.key(1), state, 0, env.default_params)
```

`step` **automatically resets** on termination. Do not call `reset` again when
`done` is true. For transition tests or terminal-board inspection, use
`step_env`, which does not auto-reset. Observation reconstruction is available
through `get_obs(state, params=None)`.

| Module / class | Default observation | Actions | Episode end |
|---|---|---|---|
| `battleship.Battleship` | last hit bit + legal-action mask | row-major cell | all ship cells hit |
| `pocman.PocMan` | 11 wall/food/ghost/power-pill sensor values | four directions | death, cleared pellets, or 1,000 steps |
| `minesweeper.MineSweeper` | last clue one-hot, including reset marker | row-major cell | mine, cleared safe cells, or safe-cell-count steps |
| `tmaze.TMaze` | initial cue, corridor, junction one-hot | north, south, east, west | choosing a junction branch |

## Battleship

Ships are placed sequentially. Each ship's orientation is uniform, then its
anchor is uniform among nonoverlapping placements in that orientation. Complete
boards are therefore not equally likely. Publication configurations require
positive lengths with their sum at most the smaller board dimension, ensuring
placement remains possible in either orientation.

Ordinary steps give −1, and clearing the board gives `rows × cols`, so total
return is `rows × cols + 1 - number_of_shots`. The environment's observation
tail contains the legal-action mask; the experiment strips it from the network
input and applies it to the categorical policy separately. `last_hit_miss` is
the outcome of the previous action. Hidden boards are only used by diagnostics.

## PocMan

The environment adapts Jumanji's PacMan engine to Gymnax and the partial sensor
observation. The map and line-of-sight construction follow Allen et al. Player
coordinates are `(row, column)` while ghost/pellet arrays are `(column, row)`.
Eaten pellets occupy an unreachable `(0,0)` sentinel slot.

The standalone extraction's wall-below sensor correction is included. The
sensor reads `grid[row + 1, column]`, with boundary clipping. Position and the
remaining-pellet map are not policy inputs. Importing this module requires
Jumanji; the other environments do not import it.

## Minesweeper

This is an independent implementation of POPGym's partial Minesweeper rules.
For N cells and S safe cells, a new safe query gives `+1/S`, a repeated safe
query gives `-0.5/(S-2)`, and a mine gives `-0.5-1/S`. At least three safe cells
are required. A win gives total return 1; there is no flood fill, flag action,
action mask, or first-click protection.

Observations encode the last clue with `min(8, num_mines)+1` clue bins plus a
distinct reset bin. No position is supplied; the experiment adds previous-action
input to the recurrent projection. Unlike POPGym's integer/two-coordinate
interface, observations are one-hot and actions are flat indices. The public
auto-reset interface combines termination and timeout into `done`; both reasons
are preserved in `info`. Neighbour counts exclude the centre even at a mine,
which affects only low-level terminal mine observations, replaced by reset
observations through `step`.

## T-maze

The initial cue identifies the correct junction branch. Corridor position and
goal direction are then aliased. Default good/bad rewards are 4 and −0.1.
Non-junction north/south actions are no-ops; east/west move along the corridor.
The exact experiments construct a matching tabular model and independently
verify every nonterminal state/action transition against this environment.

## Provenance

Battleship, PocMan and T-maze are adapted from Allen et al.'s
[λ-discrepancy repository](https://github.com/brownirl/lambda_discrepancy), via
the standalone `lambda-envs` extraction at revision
`dbd9dfc069430b3c4c1694bf18b8e3b14eda18df`. These sources are Apache-2.0.
PocMan also derives its step adapter from InstaDeep's Apache-2.0 Jumanji.
See the repository's `THIRD_PARTY_NOTICES.md` and `licenses/Apache-2.0.txt`,
also included under `lambda_imitation/` in the built wheel.
