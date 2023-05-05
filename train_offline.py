import os
from typing import Tuple
from pathlib import Path
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from dataset_utils import Log

from dataset_utils import MergeExpertUnion, load_d4rl_data, add_expert2suboptimal, normalize_dataset
from evaluation import evaluate_disc
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('expert_dataset_name', 'expert-v2', 'name of expert dataset')
flags.DEFINE_integer('expert_dataset_num', 1, 'num of expert dataset')
flags.DEFINE_multi_string('suboptimal_dataset_name', ['expert-v2', 'random-v2'], 'list of name of suboptimal dataset')
flags.DEFINE_multi_integer('suboptimal_dataset_num', [100, 400], 'list of num of suboptimal dataset')

flags.DEFINE_integer('seed', 77, 'Random seed.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_dataset(env_name: str, dataset_info: list) -> Tuple[MergeExpertUnion, MergeExpertUnion]:
    dataset_dir = 'dataset'
    
    # make dataset for train
    expert_info, suboptimal_info = dataset_info
    expert_dataset = load_d4rl_data(dataset_dir, env_name, expert_info, start_idx=0)
    start_idx = [expert_info[1], 0] if (expert_info[0] == suboptimal_info[0][0]) else [0,0]
    suboptimal_dataset = load_d4rl_data(dataset_dir, env_name, suboptimal_info, start_idx=start_idx)
    union_dataset = add_expert2suboptimal(suboptimal_dataset, expert_dataset)
    normalize_dataset(expert_dataset, suboptimal_dataset, union_dataset)
    train_dataset = MergeExpertUnion(expert_dataset, union_dataset)
    
    # make dataset for validation
    val_expert_info = ['expert-v2', 50]
    val_random_info = ['random-v2', 50]
    val_expert_start_idx = start_idx[0] + 1
    val_random_start_idx = start_idx[1] + 1
    val_expert_dataset = load_d4rl_data(dataset_dir, env_name, val_expert_info, start_idx=val_expert_start_idx)
    val_random_dataset = load_d4rl_data(dataset_dir, env_name, val_random_info, start_idx=val_random_start_idx)
    normalize_dataset(val_expert_dataset, suboptimal_dataset, val_random_dataset)
    val_dataset = MergeExpertUnion(val_expert_dataset, val_random_dataset)

    return train_dataset, val_dataset


def main(_):

    # make train_dataset and val_dataset
    expert_info = [FLAGS.expert_dataset_name, FLAGS.expert_dataset_num]
    suboptimal_info = [FLAGS.suboptimal_dataset_name, FLAGS.suboptimal_dataset_num]
    dataset_info = (expert_info, suboptimal_info)
    train_dataset, val_dataset = make_dataset(FLAGS.env_name, dataset_info)

    kwargs = dict(FLAGS.config)
    
    observation_example = train_dataset.expert_observations[0]
    action_example = train_dataset.expert_actions[0]
    agent = Learner(FLAGS.seed,
                    observation_example[np.newaxis],
                    action_example[np.newaxis],
                    **kwargs)
    kwargs['seed'] = FLAGS.seed
    kwargs['env_name'] = FLAGS.env_name

    log = Log(Path('benchmark')/FLAGS.env_name/f"e{FLAGS.suboptimal_dataset_num[0]}r{FLAGS.suboptimal_dataset_num[1]}", kwargs)
    log(f'Log dir: {log.dir}')
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = train_dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.eval_interval == 0:
            expert_acc, random_acc = evaluate_disc(agent, val_dataset)
            log.row({'val_expert_acc': expert_acc, 'val_random_acc': random_acc})

if __name__ == '__main__':
    app.run(main)
