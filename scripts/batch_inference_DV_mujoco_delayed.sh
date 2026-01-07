#!/bin/bash
# Usage: ./batch_inference_DV_mujoco_delayed.sh <seed> <task> <tmp> <gamma> <tau> <wf> <aa> <na>

# 任务列表（可根据需要调整）

tasks=("hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2")

# 检查点列表
pds=("2" "4" "8")

# 参数传递
seed=$1
task=$2
tmp=$3
gamma=$4
tau=$5
wf=$6
aa=$7
na=$8

# 循环执行任务
for task in "${tasks[@]}"; do
  for pd in "${pds[@]}"; do
    echo "Running task=$task pd=$pd"
    python pipelines/veteran_d4rl_mujoco.py \
      mode="inference" \
      seed="${seed}" \
      task="${task}" \
      delay="${delay}" \
      task.planner_horizon="${ph}" \
      planner_depth="${pd}" \
      noise_beta="${nb}" \
      task.planner_temperature="${tmp}" \
      num_envs=50 \
      num_episodes=10 \
      planner_num_candidates=50 \
      normalize_action=${na} \
      action_arctanh=${aa} \
      task.discount="${gamma}" \
      task.iql_tau="${tau}" \
      weight_factor="${wf}" \
      print_interval=500
  done
done


# python pipelines/veteran_d4rl_mujoco.py mode="inference" task="walker2d-medium-replay-v2" seed=1 delay=16 planner_depth=4 noise_beta=1 task.planner_temperature=1 num_envs=30 planner_num_candidates=20 planner_ckpt=1000000 filter_short=0 loss_weight_type=1