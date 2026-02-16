import gym
from cs285.networks.policies import MLPPolicy
from cs285.infrastructure import pytorch_util as ptu
from gym.spaces import Box, Discrete

env_id = 'Hopper-v4'
env = gym.make(env_id)
if isinstance(env.action_space, Discrete): 
    ac_dim, ob_dim = env.action_space.n, env.observation_space.shape[0]
    discrete = True
else: 
    ac_dim, ob_dim = env.action_space.shape[0], env.observation_space.shape[0]
    discrete = False

policy = MLPPolicy(ac_dim=ac_dim,
                   ob_dim=ob_dim,
                   discrete=discrete,
                   n_layers=2, 
                   layer_size=32, 
                   learning_rate=1e-3)

ob = env.reset()
action = policy.get_action(ob)
print(action)
