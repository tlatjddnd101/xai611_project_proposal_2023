#!/bin/bash

ENV_LIST=(
	"HalfCheetah-v2"
	"Hopper-v2"
	"Walker2d-v2"
	"Ant-v2"
)

let "gpu=0"
for env in ${ENV_LIST[@]}; do
	CUDA_VISIBLE_DEVICES=$gpu python train_offline.py \
		--env_name $env \
		--suboptimal_dataset_name 'expert-v2' \
		--suboptimal_dataset_name 'random-v2' \
		--suboptimal_dataset_num 100 \
		--suboptimal_dataset_num 400 \
		--config=config.py \
		--max_steps 100000 \
		--eval_interval 10000 \
		--seed 77 
	sleep 2
done