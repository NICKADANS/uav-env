from model import Critic, Actor
import torch
from copy import deepcopy
from replay_buffer import ReplayBuffer
from torch.optim import Adam
import torch.nn as nn
import numpy as np


# 软更新target网络参数
def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


# 硬更新(同步)target网络参数
def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


# MADDPG算法
class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size, capacity, episodes_before_train):
        # 初始化 2 * n_agents 个网络
        self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]
        self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:  # 如果可以使用GPU加速计算
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.episodes_before_train = episodes_before_train
        self.GAMMA = 0.95
        self.tau = 0.01
        self.scale_reward = 0.01

        # 优化器
        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            state_batch = torch.stack(states).type(FloatTensor)
            action_batch = torch.stack(actions).type(FloatTensor)
            reward_batch = torch.stack(rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in next_states if s is not None]).type(FloatTensor)

            # 这句话啥意思
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, next_states)))

            # for current agent
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            self.critic_optimizer[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)

            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0, 1).contiguous())

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), non_final_next_actions.view(-1, self.n_agents * self.n_actions)
            ).squeeze()

            # scale_reward: to scale reward in Q functions
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1) * self.scale_reward)
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()

            self.critic_optimizer[agent].step()
            self.actor_optimizer[agent].zero_grad()

            # 这段没看懂
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    # 挑选行为
    def select_action(self, state_batch):
        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            # 生成两个噪音
            act += torch.from_numpy(20 * np.random.randn(self.dim_act) * self.var[i]).type(FloatTensor)
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.01:
                self.var[i] *= 0.998
            # 将act的区间夹紧在 [-vmax, vmax]之间
            act = torch.clamp(act, -20, 20)
            actions[i, :] = act
        self.steps_done += 1
        return actions

    # 保存模型
    def save_model(self):
        for i in range(len(self.critics)):
            torch.save(self.critics[i].state_dict(), 'critic' + str(i) + ".pth")
            torch.save(self.actors[i].state_dict(), 'actor' + str(i) + ".pth")

    # 加载模型
    def load_model(self, path):
        if torch.cuda.is_available():
            for i in range(len(self.critics)):
                self.critics[i].load_state_dict(torch.load(path + 'critic' + str(i) + '.pth'))
                self.actors[i].load_state_dict(torch.load(path + 'actor' + str(i) + '.pth'))
        else:
            for i in range(len(self.critics)):
                self.critics[i].load_state_dict(torch.load(path + 'critic' + str(i) + '.pth', map_location=torch.device('cpu')))
                self.actors[i].load_state_dict(torch.load(path + 'actor' + str(i) + '.pth', map_location=torch.device('cpu')))