# Lambda Discrepancy based Offline Imitation Learning

This is the repository for lambda-discrepancy based offline imitation learning. 

Planned features:
 - [ ] Recording Wrapper for TD(0) and TD(1) training, as well as hidden-state recording and sampling
 - [ ] IQLearn utilizing lambda-discrepancy
 - [ ] Evaluation on simulator environments
 - [ ] Evaluation on Deep Racers

## Installation

This repo is strucured as a python package, so install using
```bash
pip install -e .
```
in the root directory of this project. (`-e` optional for quicker updates during development)

Tested for `python 3.12`.

## Tests

Tests are written using `pytest`, so run
```bash
python -m pytest
```
in the root-directory of this project.
