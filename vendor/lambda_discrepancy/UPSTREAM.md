# Minimal Battleship environment dependency

Source: https://github.com/brownirl/lambda_discrepancy
Commit: `e0c027df237915c7da28c13e9426554f977486bd`
License: Apache-2.0; complete upstream LICENSE retained alongside this notice.

Only `lamb/envs/battleship.py` is needed by the extracted trainer. It is copied
byte-for-byte from the verified study vendor tree, originally acquired at the
pinned public commit. It imports Gymnax directly; no upstream package initializer,
PPO trainer or other environment is executed. Adaptation lives outside this file.

SHA-256:
- `lamb/envs/battleship.py`: `db597c2373931508ae063f45c779fbdef5fdd1e657c833b1aa890a141e9f7818`
- `LICENSE`: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
