import torch
import gym
import cs285.infrastructure.pytorch_util as ptu
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from cs285.infrastructure.utils import sample_trajectory

env_id = 'HalfCheetah-v4'
expert_policy_file = '../policies/experts/HalfCheetah-v4')

env = gym.make(env_id)
policy = LoadedGaussianPolicy(expert_policy_file)

trajectory = sample_trajectory(env, policy, 100)
print(trajectory)
