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
import sys
sys.path.append('..')
from compare import local_greedy


REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32

class DeepQTable:
    def __init__(self, env, obs_shape, n_actions, n_agents, device="cpu"):
        self.env = env
        self.device = device
        self.n_agents = n_agents
        self.nets = [DQN(obs_shape, n_actions).to(device) for _ in range(n_agents)]
        self.tgt_nets = [DQN(obs_shape, n_actions).to(device) for _ in range(n_agents)]
        self.buffer = [ExperienceBuffer(REPLAY_SIZE) for _ in range(n_agents)]
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
    def play_step(self, epsilon=0.0, render=False, gdqn=False):
        save_exp = [True for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            if self.env.uavs[i].energy == 0:
                save_exp[i] = False
        actions = self.select_actions(self.state, epsilon)
        if gdqn is True:
            greedy_actions = local_greedy.select_actions(self.env)
            if self.env.cal_reward(actions) < self.env.cal_reward(greedy_actions) + 0.01:
                actions = greedy_actions
        if render is True:
            cv2.imshow("env", self.env.render.image)
            print(actions, self.env.uavs[0].energy)
            cv2.waitKey(0)
        # do step in the environment
        new_obs, reward, dones, _ = self.env.step(actions)
        for i in range(self.n_agents):
            if save_exp[i] is True:
                exp = Experience(self.state[i], actions[i], reward[i], dones[i], new_obs[i])
                self.buffer[i].append(exp)
        self.total_reward += np.sum(reward)
        self.state = new_obs
        gameover = True
        for d in dones:
            if d == 0:
                gameover = False
                break
        done_reward = None
        if gameover:
            done_reward = self.total_reward
            print("poi_done: ", self.env.poi_done)
            self._reset()
        return done_reward, gameover

    def optimize(self):
        for i in range(self.n_agents):
            self.optimizers[i].zero_grad()
            # 计算Target-Q和Q网络的误差
            batch = self.buffer[i].sample(BATCH_SIZE)
            states, actions, rewards, dones, next_states = batch
            states_v = torch.tensor(np.array(states, copy=False)).to(self.device)
            next_states_v = torch.tensor(np.array(next_states, copy=False)).to(self.device)
            actions_v = torch.tensor(actions).type(torch.int64)
            actions_v = actions_v.to(self.device)
            rewards_v = torch.tensor(rewards).to(self.device)
            done_mask = torch.LongTensor(dones).to(self.device)
            state_action_values = self.nets[i](states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_state_values = self.tgt_nets[i](next_states_v).max(1)[0]
                for _ in range(len(next_state_values)):
                    if done_mask[_] == 1:
                        next_state_values[_] = 0.0
                next_state_values = next_state_values.detach()
            expected_state_action_values = next_state_values * GAMMA + rewards_v
            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values)
            loss_t.backward()
            # 更新神经网络
            self.optimizers[i].step()

    def hard_update(self):
        for i in range(self.n_agents):
            self.tgt_nets[i].load_state_dict(self.nets[i].state_dict())

    def save_models(self, reward):
        for i in range(self.n_agents):
            torch.save(self.nets[i], './models/' + str(reward) + '_' + str(i) + '.pt')

    def load_models(self, reward):
        for i in range(self.n_agents):
            self.nets[i].load_state_dict(torch.load('./models/' + str(reward) + '_' + str(i) + '.pt', map_location=torch.device('cpu')).cpu().state_dict())
        self.hard_update()