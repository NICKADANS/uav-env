from copy import deepcopy

import torch as T
from model import Actor, Critic


class Agent:
    def __init__(self, obs_dim, act_dim, n_agents):
        self.tau = 0.01
        self.epsilon = 1
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.critic = Critic(n_agents, obs_dim, act_dim)
        self.actor = Actor(obs_dim, act_dim)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)

    def soft_update(self):
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
        for target_param, source_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def choose_action(self, obs):
        state = T.tensor(obs, dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state)
        noise = self.epsilon * 20 * T.randn(self.act_dim).to(self.actor.device)
        action += noise
        action = T.clamp(action, -20.0, 20.0)
        return action.detach().cpu().numpy()