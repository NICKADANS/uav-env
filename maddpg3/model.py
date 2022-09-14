import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, n_agents, obs_dim, act_dim, learning_rate=0.001):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear((obs_dim + act_dim) * n_agents, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs, act):
        x = T.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, learning_rate=0.0001):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, act_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return T.softmax(self.fc3(x), dim=0)