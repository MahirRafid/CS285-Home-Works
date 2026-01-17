import torch
import torch.nn as nn
import gym
import cs285.infrastructure.pytorch_util as ptu
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy as LGP
import cs285.infrastructure.utils as utils

env_id = 'HalfCheetah-v4'
# vs_code_policy_file = '/Users/mahir05/Desktop/RL/CS285/CS285-Home-Works/hw1/cs285/policies/experts/HalfCheetah.pkl'
expert_policy_file = '/Users/mahir05/Desktop/RL/CS285/CS285-Home-Works/hw1/cs285/policies/experts/HalfCheetah.pkl'

env = gym.make(env_id)
actor = MLPPolicySL(6, 17, 2, 32)
expert_policy = LGP(expert_policy_file)

paths, _ = utils.sample_trajectories(env, actor, 10, 5)

for path in paths: 
    print(type(path['action']))
    path['action'] = expert_policy.forward(ptu.from_numpy(path['observation']))
    print(type(path['action']))
