import torch
import gym
import cs285.infrastructure.pytorch_util as ptu
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import cs285.infrastructure.utils as utils

env_id = 'HalfCheetah-v4'
expert_policy_file = '../policies/experts/HalfCheetah.pkl'

env = gym.make(env_id)
policy = LoadedGaussianPolicy(expert_policy_file)

transitions = utils.sample_trajectory(env, policy, 100)
paths, steps = transitions, len(transitions['terminal'])
print(paths, steps)
