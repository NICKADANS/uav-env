import argparse
import time
from model import DQN
import cv2
import numpy as np
import collections
from buffer import Experience
import torch
import torch.nn as nn
import torch.optim as optim
from buffer import ExperienceBuffer



REPLAY_SIZE = 50000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

GAMMA = 0.99
BUFFER_MAX_SIZE = 50000
BATCH_SIZE = 32

class DeepQTable:
    def __init__(self, env, obs_shape, n_actions, n_agents, device="cpu"):
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.nets = [DQN(obs_shape, n_actions).to(device) for _ in range(n_agents)]
        self.tgt_nets = [DQN(obs_shape, n_actions).to(device) for _ in range(n_agents)]
        self.buffer = ExperienceBuffer(REPLAY_SIZE)
        self.optimizers = [optim.Adam(net.parameters(), lr=LEARNING_RATE) for net in self.nets]
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def select_actions(self, obs, epsilon):
        actions = []
        for i in range(self.n_agents):
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                state_a = np.array([obs[i]], copy=False)
                state_v = torch.tensor(state_a).to(device=self.device)
                state_v = state_v.to(torch.float32)
                q_vals_v = self.nets[i](state_v)
                _, act_v = torch.max(q_vals_v, dim=1)
                action = int(act_v.item())
            actions.append(action)
        return actions

    @torch.no_grad()
    def play_step(self, epsilon=0.0):
        actions = self.select_actions(self.state, epsilon)
        # do step in the environment
        new_obs, reward, dones, _ = self.env.step(actions)
        for i in range(self.n_agents):
            exp = Experience(self.state[i], actions[i], reward[i], dones[i], new_obs[i])
            self.buffer.append(exp)
        self.total_reward += reward[0]
        self.state = new_obs
        gameover = True
        for d in dones:
            if d == 0:
                gameover = False
                break
        done_reward = None
        if gameover:
            done_reward = self.total_reward
            self._reset()
        return done_reward, gameover

    def optimize(self):
        for i in range(self.n_agents):
            self.optimizers[i].zero_grad()
            # 计算Target-Q和Q网络的误差
            batch = self.buffer.sample(BATCH_SIZE)
            states, actions, rewards, dones, next_states = batch
            states_v = torch.tensor(np.array(states, copy=False)).to(self.device)
            next_states_v = torch.tensor(np.array(next_states, copy=False)).to(self.device)
            actions_v = torch.tensor(actions).to(self.device)
            rewards_v = torch.tensor(rewards).to(self.device)
            done_mask = torch.tensor(dones).to(self.device)
            state_action_values = self.nets[i](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_state_values = self.tgt_nets[i](next_states_v).max(1)[0]
                next_state_values[done_mask] = 0.0
                next_state_values = next_state_values.detach()
            expected_state_action_values = next_state_values * GAMMA + rewards_v
            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
            loss_t.backward()
            # 更新神经网络
            self.optimizers[i].step()

    def hard_update(self):
        for i in range(self.n_agents):
            self.tgt_nets[i].load_state_dict(self.nets[i].state_dict())