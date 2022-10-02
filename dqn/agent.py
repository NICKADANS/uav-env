import argparse
import time

import cv2
import numpy as np
import collections
from buffer import Experience
import torch
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()[0]
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            # cv2.imshow("s", np.transpose(state_a[0], (1, 2, 0)))
            # cv2.waitKey(0)

            state_v = torch.tensor(state_a).to(device=device)
            state_v = state_v.to(torch.float32)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step([action])
        self.total_reward += reward[0]

        exp = Experience(self.state, action, reward[0], is_done[0], new_state[0])
        self.exp_buffer.append(exp)
        self.state = new_state[0]
        if is_done[0] == 1:
            done_reward = self.total_reward
            self._reset()
        return done_reward, is_done
