from typing import Dict

import flax.linen as nn
import gym
import numpy as np
import d4rl

from collections import deque
import random

def il_evaluate(env, agent, num_episodes):

    def get_normalized_score(env_name, score):
        if 'halfcheetah' in env_name.lower():
            min_score = -288.55
            max_score = 10645.07
        elif 'hopper' in env_name.lower():
            min_score = 17.89
            max_score = 3507.33
        elif 'walker2d' in env_name.lower():
            min_score = 1.98
            max_score = 4915.20
        elif 'ant' in env_name.lower():
            min_score = -57.73
            max_score = 4616.29
        
        return (score - min_score) / (max_score - min_score) * 100
    
    total_returns = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if 'ant' in env.spec.id.lower():
                state = np.concatenate((state[:27], [0.]), -1)
                
            action = agent.sample_actions(state, temperature=0.0)
            state, reward, done, _ = env.step(action)            
            total_returns += reward
            
    mean_score = total_returns / num_episodes
    normalized_mean_score = get_normalized_score(env.spec.id, mean_score)

    return normalized_mean_score

def evaluate_disc(agent, val_dataset):
    
    acc = None
    return acc