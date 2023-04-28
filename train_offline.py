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
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')

flags.DEFINE_string('expert_dataset_name', 'expert-v2', 'name of expert dataset')
flags.DEFINE_string('expert_dataset_num', 1, 'num of expert dataset')
flags.DEFINE_multi_string('suboptimal_dataset_name', ['expert-v2', 'random-v2'], 'list of name of suboptimal dataset')
flags.DEFINE_multi_integer('suboptimal_dataset_num', [400, 100], 'list of num of suboptimal dataset')

flags.DEFINE_string('save_dir', './results/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_string('mix_dataset', 'None', 'mix the dataset')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_string('alg', 'SQL', 'the training algorithm')
flags.DEFINE_float('alpha', 1.0 , 'temperature')
flags.DEFINE_float('cost_grad_coeff', 10.0 , 'cost gradient penalty coefficient')
flags.DEFINE_float('grad_coeff', 1e-4 , 'v and q gradient penalty coefficient')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        # pass
        normalize(dataset)

    return env, dataset

def make_env_and_imitation_dataset(env_name: str, dataset_info: list, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    dataset_dir = 'dataset'
    
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)
    
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    expert_info, suboptimal_info = dataset_info
    expert_dataset = load_d4rl_data(dataset_dir, env_name, expert_info, start_idx=0)
    start_idx = [expert_info[1], 0] if (expert_info[0] == suboptimal_info[0][0]) else [0,0]
    suboptimal_dataset = load_d4rl_data(dataset_dir, env_name, suboptimal_info, start_idx=start_idx)
    union_dataset = add_expert2suboptimal(suboptimal_dataset, expert_dataset)
    
    imitation_dataset = MergeExpertUnion(expert_dataset, union_dataset)
    
    return env, imitation_dataset


def main(_):
    # env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)
    expert_info = [FLAGS.expert_dataset_name, FLAGS.expert_dataset_num]
    suboptimal_info = [FLAGS.suboptimal_dataset_name, FLAGS.suboptimal_dataset_num]
    dataset_info = (expert_info, suboptimal_info)
    env, dataset = make_env_and_imitation_dataset(FLAGS.env_name, dataset_info, FLAGS.seed)
    kwargs = dict(FLAGS.config)
    kwargs['alpha'] = FLAGS.alpha
    kwargs['alg'] = FLAGS.alg
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    **kwargs)
    kwargs['seed'] = FLAGS.seed
    kwargs['env_name'] = FLAGS.env_name

    wandb.init(
        project='test',
        entity='ssw030830',
        name=f"{FLAGS.env_name}",
        config=kwargs
    )

    log = Log(Path('benchmark')/FLAGS.env_name, kwargs)
    log(f'Log dir: {log.dir}')
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            wandb.log(update_info, i)

        if i % FLAGS.eval_interval == 0:
            normalized_return = evaluate(FLAGS.env_name, agent, env, FLAGS.eval_episodes)
            log.row({'normalized_return': normalized_return})
            wandb.log({'normalized_return': normalized_return}, i)


if __name__ == '__main__':
    app.run(main)
