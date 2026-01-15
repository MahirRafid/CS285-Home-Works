import torch
import gym
import cs285.infrastructure.pytorch_util as ptu
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import cs285.infrastructure.utils as utils

env_id = 'HalfCheetah-v4'
expert_policy_file = '../policies/experts/HalfCheetah.pkl'

env = gym.make(env_id)
policy = LoadedGaussianPolicy(expert_policy_file)

paths, steps = utils.sample_trajectories(env, policy, 1000, 100)

print(len(paths))
