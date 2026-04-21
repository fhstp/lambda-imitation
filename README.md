
# Lambda Imitation

## Roadmap

 - [ ] WP1: Buffer
    - [x] 1.1: Observations, actions, rewards, hidden states, behavior probabilities, termination, sampling_ok
        - Note: buffer was modified to take any key/shape pair, so all these are supported, but not necessarily implemented!
    - [ ] 1.2 (1.1): returns (soft and hard), importance factors, hidden state recalculations
    - [ ] 1.3 (1.1): truncation handling through sampling_ok
        - Note: not really relevant any more, the gymnax API does not support truncation
 - [ ] WP2: SAC
    - [x] 2.1 (1.3): TD Q-value approximation for given policy
        - [x] discrete
        - [x] continuous
    - [ ] 2.2 (1): MC Q-value approximation for given policy
        - [ ] discrete
        - [ ] continuous
    - [x] 2.3 (1, 2.1): soft Q-values for TD
        - [x] discrete
        - [x] continuous
    - [ ] 2.4 (1, 2.2): soft Q-values for MC
        - [ ] discrete
        - [ ] continuous
    - [ ] 2.5 (2.3, 2.4): V calculation
        - [ ] discrete
        - [ ] continuous
    - [ ] 2.6 (1): LSTM feature extraction
    - [ ] 2.7 (2.5): lambda discrepancy
    - [ ] 2.8 (2.6, 2.7): train LSTM feature extraction through lambda discrepancy
    - [ ] 2.9 (2.5): policy improvement
        - [ ] discrete
        - [ ] continuous
    - [x] 2.10 (2.9): automatic alpha tuning
    - [ ] 2.11: logging of metrics in wandb
 - [ ] WP3: IQLearn
    - [ ] 3.1 (2.5): reward training
    - [ ] 3.2 (3.1): logging of additional metrics

Evaluation Environments:
 - Discrete and Continuous Gridworlds, where Q-values are explicitly calculatable
