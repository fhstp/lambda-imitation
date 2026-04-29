#/bin/bash


URANDOM_SEED=$(od -vAn -N4 -tu4 < /dev/urandom | tr -d ' ')

MY_PID=$$

COMBINED_SEED=$((URANDOM_SEED + MY_PID))

./.conda/envs/lambda/bin/python ./git/lambda-imitation/examples/cartpole_sac_mc.py --wandb --rounds 500 --partial --wandb-project "lambda-imitaiton-cartpole-sanity"   --lambda-coef 0.0 --tau 0.005 --seed $COMBINED_SEED
