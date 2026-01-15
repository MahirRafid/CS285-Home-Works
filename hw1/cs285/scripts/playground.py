import torch
import gym
import cs285.infrastructure.pytorch_util as ptu
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import cs285.infrastructure.utils as utils

env_id = 'HalfCheetah-v4'
expert_policy_file = '../policies/experts/HalfCheetah.pkl'

env = gym.make(env_id)
actor = MLPPolicySL(6, 17, 2, 16)
print(actor(env.reset()))
expert_policy = LoadedGaussianPolicy(expert_policy_file)

paths = utils.sample_trajectory(env, actor, 10)
print(paths)

