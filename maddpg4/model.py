import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_shape, act_size):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out(obs_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, act_size),
            nn.Tanh()
        )

    def get_conv_out(self, shape):
        x = self.conv(torch.zeros(1, *shape))
        return int(np.prod(x.size()))

    def forward(self, state):
        x = self.conv(state)
        x = self.fc(x.view(x.size()[0], -1))
        return x


class Critic(nn.Module):
    def __init__(self, obs_shape, act_size):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_out(obs_shape)
        self.fc1 = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
           nn.Linear(1024 + act_size, 512),
           nn.ReLU(),
           nn.Linear(512, 1)
        )

    def get_conv_out(self, shape):
        x = self.conv(torch.zeros(1, *shape))
        return int(np.prod(x.size()))

    def forward(self, s, a):
        x = self.conv(s)
        x = self.fc1(x.view(x.size()[0], -1))
        x = self.fc2(torch.cat([x, a], dim=1))
        return x
