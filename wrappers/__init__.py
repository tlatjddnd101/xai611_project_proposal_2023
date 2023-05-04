import gym
import numpy as np

from wrappers.episode_monitor import EpisodeMonitor
from wrappers.single_precision import SinglePrecision
from wrappers.absorbing_wrapper import AbsorbingWrapper
from wrappers.normalize_state_wrapper import NormalizeStateWrapper

def normalize_env(env_name, expert_dataset, suboptimal_dataset, union_dataset, seed):
    
    shift = -np.mean(suboptimal_dataset.observations, 0)
    scale = 1.0 / (np.std(suboptimal_dataset.observations, 0) + 1e-3)
    
    # normalize
    union_dataset.init_observations = (union_dataset.init_observations + shift) * scale
    expert_dataset.observations = (expert_dataset.observations + shift) * scale
    expert_dataset.next_observations = (expert_dataset.next_observations + shift) * scale
    union_dataset.observations = (union_dataset.observations + shift) * scale
    union_dataset.next_observations = (union_dataset.next_observations + shift) * scale
    
    # ignore absorbing state
    union_dataset.init_observations = np.c_[union_dataset.init_observations, np.zeros(len(union_dataset.init_observations), dtype=np.float32)]
    expert_dataset.observations = np.c_[expert_dataset.observations, np.zeros(len(expert_dataset.observations), dtype=np.float32)]
    expert_dataset.next_observations = np.c_[expert_dataset.next_observations, np.zeros(len(expert_dataset.next_observations), dtype=np.float32)]
    union_dataset.observations = np.c_[union_dataset.observations, np.zeros(len(union_dataset.observations), dtype=np.float32)]
    union_dataset.next_observations = np.c_[union_dataset.next_observations, np.zeros(len(union_dataset.next_observations), dtype=np.float32)]
    
    if 'ant' in env_name.lower():
        shift = np.concatenate((shift, np.zeros(84)))
        scale = np.concatenate((scale, np.ones(84)))
        
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    env.seed(seed)
    eval_env.seed(seed+1)
    
    env = NormalizeStateWrapper(env, shift=shift, scale=scale)
    eval_env = NormalizeStateWrapper(eval_env, shift=shift, scale=scale)

    return AbsorbingWrapper(env), AbsorbingWrapper(eval_env)