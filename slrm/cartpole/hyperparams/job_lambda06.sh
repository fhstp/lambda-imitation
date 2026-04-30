#/bin/bash

run_on_gpu() {
    local gpu_id=$1
    local urand
    urand=$(od -vAn -N4 -tu4 < /dev/urandom | tr -d ' ')
    local seed=$((urand + $$ + gpu_id))
    CUDA_VISIBLE_DEVICES=$gpu_id ./.conda/envs/jax/bin/python ./git/lambda-imitation/examples/cartpole_sac_mc.py --wandb --rounds 500 --partial --wandb-project "lambda-imitaiton-cartpole-sanity" --lambda-coef 0.6 --tau 0.005 --seed $seed
}

run_on_gpu 0 &
PID_0=$!
run_on_gpu 1 &
PID_1=$!

wait $PID_0; EXIT_0=$?
wait $PID_1; EXIT_1=$?

if [ $EXIT_0 -ne 0 ] || [ $EXIT_1 -ne 0 ]; then
    echo "process exit codes: gpu0=$EXIT_0 gpu1=$EXIT_1"
    exit 1
fi
