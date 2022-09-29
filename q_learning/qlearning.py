import random
from copy import deepcopy

import numpy as np
from uav_env import UavEnvironment

class QLearningTable:
    # 初始化
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度
        self.q_table = {}
        self.action_space = action_space

    # 根据epsilon的概率在当前可行行动下随机选择下一个行动(next_index)
    def choose_action(self, state, explore=True):
        # 依据epsilon的概率，在当前state下从Q表中贪婪选择action
        if explore is False:
            # 选择Q-value最高的action执行
            max_value = -99999
            action = self.action_space[np.random.randint(0, len(self.action_space))]
            for a in self.action_space:
                value = self.check_qstate_exist((state, a))
                if max_value < value:
                    action = a
                    max_value = value
        else:
            if np.random.uniform() < self.epsilon:
                # 选择Q-value最高的action执行
                max_value = -99999
                action = self.action_space[np.random.randint(0, len(self.action_space))]
                for a in self.action_space:
                    value = self.check_qstate_exist((state, a))
                    if max_value < value:
                        action = a
                        max_value = value
            else:
                # 从当前可采取的行动中随机选择一个执行
                action = self.action_space[np.random.randint(0, len(self.action_space))]
        return action

    # 学习
    def learn(self, state, action, reward, next_state, done):
        self.check_qstate_exist((state, action))
        q_predict = self.q_table[state, action]
        if done == 1:
            q_target = reward
        else:
            future_rewards = []
            for tuple in self.q_table:
                if tuple[0] == next_state:
                    future_rewards.append(self.q_table[tuple])
            q_target = reward + self.gamma * max(future_rewards) if len(future_rewards) > 0 else 0

        self.q_table[state, action] += self.lr * (q_target - q_predict)  # update

    # 查询<状态S,行为a>是否在Q表中，不在则新建一个，在则返回其q值
    def check_qstate_exist(self, tuple):
        if self.q_table.get(tuple) is None:
            self.q_table[tuple] = 0
            return 0
        return self.q_table[tuple]

