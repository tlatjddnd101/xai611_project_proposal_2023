#!/bin/bash

NUM_GPUS=4
ENV_LIST=(
	"hopper-medium-v2"
	"halfcheetah-medium-v2"
	"walker2d-medium-v2"
	"hopper-medium-replay-v2"
	"halfcheetah-medium-replay-v2"
	"walker2d-medium-replay-v2"
	"hopper-medium-expert-v2"
	"halfcheetah-medium-expert-v2"
	"walker2d-medium-expert-v2"
) 

let "gpu=2"
for env in ${ENV_LIST[@]}; do
	XLA_PYTHON_CLIENT_MEM_FRACTION=.10 CUDA_VISIBLE_DEVICES=$gpu python train_offline.py \
		--env_name $env \
		--suboptimal_dataset_name 'expert-v2' \
		--suboptimal_dataset_name 'random-v2' \
		--suboptimal_dataset_num 400 \
		--suboptimal_dataset_num 100 \
		--config=configs/mujoco_config.py \
		--alg "EQL" \
		--alpha 2.0 \
		--max_steps 1000000 \
		--log_interval 10000 \
		--eval_interval 10000 \
		--eval_episodes 10 \
		--seed 77 &
	sleep 2
	let "gpu=(gpu+1)%$NUM_GPUS"
done