import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_size, act_size, action_lim):
        super(Actor, self).__init__()
        self.action_lim = action_lim
        self.fc1 = nn.Linear(obs_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, act_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x) * self.action_lim
        return x


class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(512 + act_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))