
# Lambda Imitation

## Roadmap

 - [ ] AP1: Buffer
    - [ ] 1.1: Observations, actions, rewards, hidden states, behavior probabilities, termination, sampling_ok
    - [ ] 1.2 (1.1): returns (soft and hard), importance factors, hidden state recalculations
    - [ ] 1.3 (1.1): truncation handling through sampling_ok
 - [ ] AP2: SAC
    - [ ] 2.1 (1.3): TD Q-value approximation for given policy
        - [ ] discrete
        - [ ] continuous
    - [ ] 2.2 (1): MC Q-value approximation for given policy
        - [ ] discrete
        - [ ] continuous
    - [ ] 2.3 (1, 2.1): soft Q-values for TD
        - [ ] discrete
        - [ ] continuous
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
    - [ ] 2.10 (2.9): automatic alpha tuning
    - [ ] 2.11: logging of metrics in wandb
 - [ ] AP3: IQLearn
    - [ ] 3.1 (2.5): reward training
    - [ ] 3.2 (3.1): logging of additional metrics

Evaluation Environments:
 - Discrete and Continuous Gridworlds, where Q-values are explicitly calculatable
