import torch

from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01
        self.scale_reward = 0.01

        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        if self.episode_done <= self.episodes_before_train:
            return

        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for agent in range(self.n_agents):
            # 提取经验
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            states_batch = th.stack(batch.states).type(FloatTensor)
            actions_batch = th.stack(batch.actions).type(FloatTensor)
            rewards_batch = th.stack(batch.rewards).type(FloatTensor)
            next_state_batch = th.stack(batch.next_states).type(FloatTensor)
            next_actions = th.stack([self.actors_target[i](next_state_batch[:, i, :]) for i in range(self.n_agents)])
            next_actions = next_actions.transpose(0, 1).contiguous()

            # 更新critic网络
            current_Q = self.critics[agent](
                states_batch.view(self.batch_size, -1), actions_batch.view(self.batch_size, -1)
            )
            q_next = self.critics_target[agent](
                next_state_batch.view(self.batch_size, -1), next_actions.view(self.batch_size, -1)
            )
            target_Q = rewards_batch[:, agent].view(self.batch_size, -1) + self.GAMMA * q_next
            critic_loss = nn.MSELoss()(current_Q, target_Q)
            self.critic_optimizer[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[agent].step()

            # 更新actor网络
            state_i = states_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            # 重新选择联合动作中当前agent的动作，其他agent的动作不变
            action_batch_clone = actions_batch.clone()
            action_batch_clone[:, agent, :] = action_i
            actor_loss = -self.critics[agent](
                states_batch.view(self.batch_size, -1), action_batch_clone.view(self.batch_size, -1)
            )
            actor_loss = actor_loss.mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

        # 更新神经网络
        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(self.n_agents, self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            act += th.from_numpy((20 * np.random.random(self.n_actions) - 10) * self.var[i]).type(FloatTensor)
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.99999
            act = th.clamp(act, -20, 20)
            actions[i, :] = act
        self.steps_done += 1
        return actions

    # 保存模型
    def save_model(self):
        for i in range(len(self.critics)):
            th.save(self.critics[i].state_dict(), 'critic' + str(i) + ".pth")
            th.save(self.actors[i].state_dict(), 'actor' + str(i) + ".pth")

    # 加载模型
    def load_model(self, path):
        if th.cuda.is_available():
            for i in range(len(self.critics)):
                self.critics[i].load_state_dict(th.load('critic' + str(i) + '.pth'))
                self.actors[i].load_state_dict(th.load('actor' + str(i) + '.pth'))
        else:
            for i in range(len(self.critics)):
                self.critics[i].load_state_dict(
                    th.load('critic' + str(i) + '.pth', map_location=th.device('cpu')))
                self.actors[i].load_state_dict(
                    th.load('actor' + str(i) + '.pth', map_location=th.device('cpu')))

        self.critics_target = deepcopy(self.critics)
        self.actors_target = deepcopy(self.actors)
