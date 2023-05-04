#!/bin/bash

ENV_LIST=(
	"HalfCheetah-v2"
	"Hopper-v2"
	"Walker2d-v2"
	"Ant-v2"
)

ALGO_LIST=(
	"zril"
	"sqla1"
	"drdemo"
	"demodice"
)

let "gpu=3"
for env in ${ENV_LIST[@]}; do
for algo in ${ALGO_LIST[@]}; do
	XLA_PYTHON_CLIENT_MEM_FRACTION=.10 CUDA_VISIBLE_DEVICES=$gpu python train_offline.py \
		--env_name $env \
		--suboptimal_dataset_name 'expert-v2' \
		--suboptimal_dataset_name 'random-v2' \
		--suboptimal_dataset_num 100 \
		--suboptimal_dataset_num 400 \
		--config=configs/mujoco_config.py \
		--alg $algo \
		--alpha 2.0 \
		--max_steps 1000000 \
		--log_interval 10000 \
		--eval_interval 10000 \
		--eval_episodes 10 \
		--seed 77 
	sleep 2
done
done