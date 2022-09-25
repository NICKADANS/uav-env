import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = dim_action * n_agent
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, 1024),
            nn.ReLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(1024 + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        x = self.obs_net(obs)
        combined = th.cat([x, acts], dim=1)
        return self.out_net(combined)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_observation, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim_action)
        )

    # action output between -2 and 2
    def forward(self, obs):
        return self.net(obs)