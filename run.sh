mkdir -p logs
stop=
trap 'stop=1; echo "^C caught — finishing current run, then exiting loop"' INT
while [ -z "$stop" ]; do
  lc=$(od -An -N8 -tu8 /dev/urandom | awk '{printf "%.4f", 0.1 + 1.9*$1/2^64}')
  ts=$(date +%Y%m%d_%H%M%S)
  echo "=== lambda_coef=$lc ==="
  python examples/lambda-envs/battleship_sac_mc.py \
      --lambda-coef "$lc" \
      --num-seeds 10 --concurrent-seeds 10 \
      --wandb --wandb-per-seed --wandb-group "lc-sweep-${ts}-lc${lc}" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
done
