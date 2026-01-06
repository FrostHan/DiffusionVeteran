#!/bin/bash
# Usage: ./batch_inference_DV_mujoco_delayed.sh {seed} {last} {delay} {pd} {ph} {nb} {tmp} {gamma} {lr} {tau} {wf}

# 任务列表（可根据需要调整）
# tasks=("halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" "halfcheetah-medium-expert-v2" "hopper-medium-expert-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" "HalfCheetah" "Hopper" "Walker2d" "Ant")
# tasks=("HalfCheetah" "Walker2d" "Ant" "Hopper")
# tasks=("hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-medium-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2" "halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2")
# tasks=("HalfCheetah" "Ant")
tasks=("walker2d-medium-v2" "walker2d-medium-replay-v2")
# tasks=("hopper-medium-expert-v2" "hopper-medium-v2")
# tasks=("halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2" "walker2d-medium-v2")

# 检查点列表
ckpts=("latest")

# 参数传递
seed=$1
last=$2
delay=$3
pd=$4
ph=$5
nb=$6
tmp=$7
gamma=$8
lr=$9
tau=${10}
wf=${11}

# 循环执行任务
for task in "${tasks[@]}"; do
  for ckpt in "${ckpts[@]}"; do
    echo "Running task=$task ckpt=$ckpt"
    python pipelines/veteran_d4rl_mujoco.py \
      mode="inference" \
      seed="${seed}" \
      task="${task}" \
      delay="${delay}" \
      task.planner_horizon="${ph}" \
      planner_depth="${pd}" \
      noise_beta="${nb}" \
      task.planner_temperature="${tmp}" \
      num_envs=20 \
      num_episodes=10 \
      planner_num_candidates=40 \
      critic_ckpt="${ckpt}" \
      planner_ckpt="latest" \
      action_normalizer_type=1 \
      only_use_last_1m_steps="${last}" \
      task.discount="${gamma}" \
      task.critic_learning_rate="${lr}" \
      task.iql_tau="${tau}" \
      weight_factor="${wf}" \
      print_interval=500
  done
done


# python pipelines/veteran_d4rl_mujoco.py mode="inference" task="walker2d-medium-replay-v2" seed=1 delay=16 planner_depth=4 noise_beta=1 task.planner_temperature=1 num_envs=30 planner_num_candidates=20 planner_ckpt=1000000 filter_short=0 loss_weight_type=1