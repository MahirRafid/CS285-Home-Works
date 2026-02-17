import gym
from cs285.networks.policies import MLPPolicy
from cs285.infrastructure import pytorch_util as ptu
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

env_id = 'CartPole-v0'
env = gym.make(env_id)
if isinstance(env.action_space, Discrete): 
    ac_dim, ob_dim = env.action_space.n, env.observation_space.shape[0]
    discrete = True
else: 
    ac_dim, ob_dim = env.action_space.shape[0], env.observation_space.shape[0]
    discrete = False

seed = 42
np.random.seed(seed=seed)
torch.manual_seed(seed=seed)

env.reset(seed=seed)
env.action_space.seed(seed=seed)
env.observation_space.seed(seed=seed)

actor_net = nn.Linear(ob_dim, ac_dim)
critic_net = nn.Linear(ob_dim, 1)
gamma = 0.99

def get_action(ob): 
    ob = ptu.from_numpy(ob)
    logits = actor_net(ob)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

states, actions, rewards, dones, log_probs, next_states = [], [], [], [], [], []
done = False
ob = env.reset()
while not done: 
    action, log_prob = get_action(ob)
    next_state, reward, done, _ = env.step(action)
    states.append(ptu.from_numpy(ob))
    actions.append(action)
    rewards.append(reward)
    dones.append(done)
    log_probs.append(log_prob)
    next_states.append(ptu.from_numpy(next_state))
    ob = next_state

states = torch.stack(states)
next_states = torch.stack(next_states)
dones = torch.FloatTensor(dones)

state_values = critic_net(states).squeeze()

next_values = critic_net(next_states).squeeze().to('cpu').detach()
q_values = torch.FloatTensor(rewards) + gamma * next_values * (1-dones)

advantages = q_values - state_values
print(f'Vanilla Advantages => \n{advantages}')

# GAE
advantages = []
lam = 0.95
for t in range(len(rewards))[::-1]: 
    delta = rewards[t] + gamma * next_values[t] * (1-dones[t]) - state_values[t]
    gae = advantages[0] if len(advantages) > 0 else 0
    advantages.insert(0, delta + gamma * lam * gae)

advantages = torch.stack(advantages)
print(f'GAE Advantages => \n{advantages}')
