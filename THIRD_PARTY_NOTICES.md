# Third-party attribution

## Allen et al.: λ-discrepancy environments and network architectures

The bundled **Battleship, PocMan, and T-maze** environments are adapted from
the implementation accompanying:

> Cameron Allen, Aaron Kirtland, Ruo Yu Tao, Sam Lobel, Daniel Scott,
> Nicholas Petrocelli, Omer Gottesman, Ronald Parr, Michael Littman, and
> George Konidaris. *Mitigating Partial Observability in Sequential Decision
> Processes via the Lambda Discrepancy*. NeurIPS, 2024.

- Upstream repository: <https://github.com/brownirl/lambda_discrepancy>
- Upstream files: `lamb/envs/battleship.py`, `lamb/envs/pocman.py`,
  `lamb/envs/tmaze.py`; network architectures in `lamb/models.py`.
- License: **Apache License 2.0**, reproduced in
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- The immediate source is the local standalone `lambda-envs` extraction,
  revision **`dbd9dfc069430b3c4c1694bf18b8e3b14eda18df`**.
- The upstream checkout inspected for attribution is revision
  `e0c027df237915c7da28c13e9426554f977486bd`. This is an attribution reference,
  not a claim that the extraction is byte-identical to that revision.

Adaptations include the Gymnax interface, last-shot Battleship observations and
action masking, and the corrected PocMan wall-below column index. Publication
cleanup bundles only the environments used here, makes `get_obs` accept optional
environment parameters, removes unused environment demos and reward/slip
variants, fixes rectangular Battleship placement bounds, and aligns T-maze's
terminal predicate with the branch-selection transition. The deterministic
T-maze and square Battleship transition/reward laws are retained. Each adapted
source file carries an attribution and modification notice.

`src/lambda_imitation/utils.py` implements the Battleship skip-connection and
PocMan Dense–ReLU architectures described by the same upstream implementation,
using Flax NNX and explicit previous-action inputs. Credit for those
architectures belongs to Allen et al.

## Jumanji / InstaDeep

PocMan uses the PacMan engine and ASCII generator from
<https://github.com/instadeepai/jumanji>. Its Gymnax `step_env` adapter derives
from Jumanji's `jumanji/environments/routing/pac_man/env.py`.
Jumanji is Copyright 2022 InstaDeep Ltd and distributed under **Apache-2.0**.
The license is included at [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
Jumanji remains an installable dependency; its engine is not copied here.

## POPGym Minesweeper rules

The Minesweeper environment is an independent JAX implementation of the rules
in <https://github.com/proroklab/popgym/blob/master/popgym/envs/minesweeper.py>.
It uses a one-hot clue observation with an explicit reset marker, flat cell
actions and Gymnax auto-reset. There is no POPGym dependency. See the
[environment README](src/lambda_imitation/envs/README.md) for the precise rules
and differences. Minesweeper is not derived from Allen et al.'s repository.
