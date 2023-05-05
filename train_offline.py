import os
from typing import Tuple
from pathlib import Path
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from dataset_utils import Log
import wandb

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories, MergeExpertUnion, load_d4rl_data, add_expert2suboptimal
from evaluation import evaluate, il_evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('expert_dataset_name', 'expert-v2', 'name of expert dataset')
flags.DEFINE_integer('expert_dataset_num', 1, 'num of expert dataset')
flags.DEFINE_multi_string('suboptimal_dataset_name', ['expert-v2', 'random-v2'], 'list of name of suboptimal dataset')
flags.DEFINE_multi_integer('suboptimal_dataset_num', [400, 100], 'list of num of suboptimal dataset')

flags.DEFINE_string('save_dir', './results/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_float('cost_grad_coeff', 10.0 , 'cost gradient penalty coefficient')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env_and_imitation_dataset(env_name: str, dataset_info: list, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    dataset_dir = 'dataset'
    
    expert_info, suboptimal_info = dataset_info
    expert_dataset = load_d4rl_data(dataset_dir, env_name, expert_info, start_idx=0)
    start_idx = [expert_info[1], 0] if (expert_info[0] == suboptimal_info[0][0]) else [0,0]
    suboptimal_dataset = load_d4rl_data(dataset_dir, env_name, suboptimal_info, start_idx=start_idx)
    union_dataset = add_expert2suboptimal(suboptimal_dataset, expert_dataset)
    
    env, eval_env = wrappers.normalize_env(env_name, expert_dataset, suboptimal_dataset, union_dataset, seed)
    imitation_dataset = MergeExpertUnion(expert_dataset, union_dataset)

    return env, eval_env, imitation_dataset


def main(_):

    # make dataset and environment
    expert_info = [FLAGS.expert_dataset_name, FLAGS.expert_dataset_num]
    suboptimal_info = [FLAGS.suboptimal_dataset_name, FLAGS.suboptimal_dataset_num]
    dataset_info = (expert_info, suboptimal_info)
    env, eval_env, dataset = make_env_and_imitation_dataset(FLAGS.env_name, dataset_info, FLAGS.seed)

    kwargs = dict(FLAGS.config)

    if 'ant' in FLAGS.env_name.lower():
        observation_example = dataset.expert_observations[0]
    else:
        observation_example = env.observation_space.sample()
    action_example = env.action_space.sample()
    agent = Learner(FLAGS.seed,
                    observation_example[np.newaxis],
                    action_example[np.newaxis],
                    **kwargs)
    kwargs['seed'] = FLAGS.seed
    kwargs['env_name'] = FLAGS.env_name

    log = Log(Path('benchmark')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)
        print('no prob')
        exit()
        update_info = agent.update(batch)

        if i % FLAGS.eval_interval == 0:
            normalized_return = il_evaluate(eval_env, agent, FLAGS.eval_episodes)

if __name__ == '__main__':
    app.run(main)
