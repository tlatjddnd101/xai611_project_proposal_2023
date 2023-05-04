import os
import collections
import h5py
from urllib import request
from sqlite3 import DatabaseError
from typing import Optional

import d4rl
# TODO use ultra only for antmaze-ultra
# import d4rlultra.d4rl as d4rl
import gym
import numpy as np
from tqdm import tqdm

import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])
ImitationBatch = collections.namedtuple(
    'ImitationBatch',
    ['union_init_observations', 'expert_observations', 'expert_actions', 'expert_next_observations', 'expert_next_actions',
     'union_observations', 'union_actions', 'union_next_observations', 'union_next_actions', 'union_dones'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class ImitationDataset(object):
    def __init__(self, init_observations: np.ndarray, observations: np.ndarray,
                 actions: np.ndarray, next_observations: np.ndarray,
                 next_actions: np.ndarray, dones_float: np.ndarray):
        self.init_observations = init_observations
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.next_actions = next_actions
        self.dones = dones_float
        

def load_d4rl_data(dirname, env_id, dataset_info, start_idx, dtype=np.float32, in_recursive=False) -> ImitationDataset:
    KEYS = ['observations', 'actions', 'rewards', 'terminals']
    MAX_EPISODE_STEPS = 1000
    dataname, num_trajectories = dataset_info
    recursive_num = -1
    if isinstance(dataname, list):
        dataname_list = dataname
        num_trajs_list = num_trajectories
        start_idx_list = start_idx
        recursive_num = len(dataname_list)
        dataname = dataname_list[0]
        num_trajectories = num_trajs_list[0]
        start_idx = start_idx_list[0]

    original_env_id = env_id
    if env_id in ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2']:
        env_id = env_id.split('-v2')[0].lower()

    filename = f'{env_id}_{dataname}'
    filepath = os.path.join(dirname, filename + '.hdf5')
    # if not exists
    if not os.path.exists(filepath):
        os.makedirs(dirname, exist_ok=True)
        # Download the dataset
        remote_url = f'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{filename}.hdf5'
        print(f'Download dataset from {remote_url} into {filepath} ...')
        request.urlretrieve(remote_url, filepath)
        print(f'Done!')

    def get_keys(h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    dataset_file = h5py.File(filepath, 'r')
    dataset_keys = KEYS
    use_timeouts = False
    use_next_obs = False
    if 'timeouts' in get_keys(dataset_file):
        if 'timeouts' not in dataset_keys:
            dataset_keys.append('timeouts')
        use_timeouts = True
    dataset = {k: dataset_file[k][:] for k in dataset_keys}
    dataset_file.close()
    N = dataset['observations'].shape[0]
    init_obs_, init_action_, obs_, action_, next_obs_, next_action_, rew_, done_ = [], [], [], [], [], [], [], []
    episode_steps = 0
    num_episodes = 0
    for i in range(N - 1):
        if 'ant' in env_id.lower():
            obs = dataset['observations'][i][:27]
            if use_next_obs:
                next_obs = dataset['next_observations'][i][:27]
            else:
                next_obs = dataset['observations'][i + 1][:27]
                next_action = dataset['actions'][i + 1][:27]
        else:
            obs = dataset['observations'][i]
            if use_next_obs:
                next_obs = dataset['next_observations'][i]
            else:
                next_obs = dataset['observations'][i + 1]
                next_action = dataset['actions'][i + 1]
        action = dataset['actions'][i]
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            is_final_timestep = dataset['timeouts'][i]
        else:
            is_final_timestep = (episode_steps == MAX_EPISODE_STEPS - 1)

        if is_final_timestep:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break
            continue

        if num_episodes >= start_idx:
            if episode_steps == 0:
                init_obs_.append(obs)
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            next_action_.append(next_action)
            done_.append(done_bool)

        episode_steps += 1
        if done_bool:
            episode_steps = 0
            num_episodes += 1
            if num_episodes >= num_trajectories + start_idx:
                break

    if recursive_num > 1:
        
        recursive_num -= 1
        dataname = dataname_list[1:]
        num_trajectories = num_trajs_list[1:]
        start_idx = start_idx_list[1:]
        dataset_info = (dataname, num_trajectories)
        np_init_obs, np_obs, np_action, np_next_obs, np_next_action, np_done = load_d4rl_data(dirname, env_id, dataset_info, 
                                                                                                start_idx, in_recursive=True)
        concat_init_obs = np.concatenate([init_obs_, np_init_obs], dtype=dtype)
        concat_obs = np.concatenate([obs_, np_obs], dtype=dtype)
        concat_action = np.concatenate([action_, np_action], dtype=dtype)
        concat_next_obs = np.concatenate([next_obs_, np_next_obs], dtype=dtype)
        concat_next_action = np.concatenate([next_action_, np_next_action], dtype=dtype)
        concat_done = np.concatenate([done_, np_done], dtype=dtype)
        
        if in_recursive:
            return concat_init_obs, concat_obs, concat_action, concat_next_obs, concat_next_action, concat_done
        else:
            print(f'{num_episodes} trajectories are sampled')
            dataset = ImitationDataset(concat_init_obs, concat_obs, concat_action, concat_next_obs, concat_next_action, concat_done)
            return dataset

    if in_recursive:
        print(f'{num_episodes} trajectories are sampled')
        return np.array(init_obs_, dtype=dtype), np.array(obs_, dtype=dtype), np.array(action_, dtype=dtype), np.array(
            next_obs_, dtype=dtype), np.array(next_action_, dtype=dtype), np.array(done_, dtype=dtype)
    else:
        print(f'{num_episodes} trajectories are sampled')
        dataset = ImitationDataset(np.array(init_obs_, dtype=dtype),
                        np.array(obs_, dtype=dtype),
                        np.array(action_, dtype=dtype),
                        np.array(next_obs_, dtype=dtype),
                        np.array(next_action_, dtype=dtype),
                        np.array(done_, dtype=dtype),
                        )

        return dataset

def add_expert2suboptimal(suboptimal_dataset: ImitationDataset, expert_dataset: ImitationDataset, dtype=np.float32) -> ImitationDataset:
    
    union_init_observations = np.concatenate([suboptimal_dataset.init_observations, expert_dataset.init_observations], dtype=dtype)
    union_observations = np.concatenate([suboptimal_dataset.observations, expert_dataset.observations], dtype=dtype)
    union_actions = np.concatenate([suboptimal_dataset.actions, expert_dataset.actions], dtype=dtype)
    union_next_observations = np.concatenate([suboptimal_dataset.next_observations, expert_dataset.next_observations], dtype=dtype)
    union_next_actions = np.concatenate([suboptimal_dataset.next_actions, expert_dataset.next_actions], dtype=dtype)
    union_dones = np.concatenate([suboptimal_dataset.dones, expert_dataset.dones], dtype=dtype)
    
    union_dataset = ImitationDataset(union_init_observations,
                                     union_observations,
                                     union_actions,
                                     union_next_observations,
                                     union_next_actions,
                                     union_dones)
    
    return union_dataset

class MergeExpertUnion(object):
    def __init__(self, expert_dataset: ImitationDataset, union_dataset: ImitationDataset):
        self.expert_init_observations = expert_dataset.init_observations
        self.expert_observations = expert_dataset.observations
        self.expert_actions = expert_dataset.actions
        self.expert_next_observations = expert_dataset.next_observations
        self.expert_next_actions = expert_dataset.next_actions
        self.expert_dones = expert_dataset.dones
        
        self.union_init_observations = union_dataset.init_observations
        self.union_observations = union_dataset.observations
        self.union_actions = union_dataset.actions
        self.union_next_observations = union_dataset.next_observations
        self.union_next_actions = union_dataset.next_actions
        self.union_dones = union_dataset.dones
    
    def sample(self, batch_size: int) -> ImitationBatch:
        union_init_indx = np.random.randint(len(self.union_init_observations), size=batch_size)
        expert_indx = np.random.randint(len(self.expert_observations), size=batch_size)
        union_indx = np.random.randint(len(self.union_observations), size=batch_size)
        return ImitationBatch(union_init_observations=self.union_init_observations[union_init_indx],
                              expert_observations=self.expert_observations[expert_indx],
                              expert_actions=self.expert_actions[expert_indx],
                              expert_next_observations=self.expert_next_observations[expert_indx],
                              expert_next_actions=self.expert_next_actions[expert_indx],
                              union_observations=self.union_observations[union_indx],
                              union_actions=self.union_actions[union_indx],
                              union_next_observations=self.union_next_observations[union_indx],
                              union_next_actions=self.union_next_actions[union_indx],
                              union_dones=self.union_dones[union_indx])


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 add_env: gym.Env='None',
                 expert_ratio: float=1.0,
                 clip_to_eps: bool = True,
                 heavy_tail: bool = False,
                 heavy_tail_higher: float = 0.,
                 eps: float = 1e-5):

        dataset = d4rl.qlearning_dataset(env)
        if add_env != 'None':
            add_data = d4rl.qlearning_dataset(add_env)
            if expert_ratio >= 1:
                raise ValueError('in the mix setting, the expert_ratio must < 1')
            length_add_data = int(add_data['rewards'].shape[0] * (1 - expert_ratio))
            length_expert_data = int(length_add_data * expert_ratio)
            for k, _ in dataset.items():
                dataset[k] = np.concatenate(
                                 [add_data[k][:-length_expert_data],
                                 dataset[k][:length_expert_data]], axis=0)
            print('-------------------------------')
            print(f'we are in the mix data regimes, len(expert):{length_expert_data} | len(add_data): {length_add_data} | expert ratio: {expert_ratio}')
            print('-------------------------------')

        if heavy_tail:
            dataset = d4rl.qlearning_dataset(env, heavy_tail=True, heavy_tail_higher=heavy_tail_higher)
        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()