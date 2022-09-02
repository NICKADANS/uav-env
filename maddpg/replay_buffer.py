# 经验缓冲区

import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """
        创建一个优先经验缓冲队列
        :param
            size: int
                Max number of transitions to store in the buffer. When the buffer
                overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    # 清空缓冲区
    def clear(self):
        self._storage = []
        self._next_idx = 0

    # 向缓冲区添加一条经验(观测值，行为，奖励，新观测值，是否完成)
    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    # 根据传入的索引数组从缓冲区里提取对应的经验
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    # 根据数量batch_size生成一批随机索引
    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    # 使用最近数量为batch_size的经验
    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    # 从缓冲区中随机取样batch_size个历史经验
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        :param
            batch_size: int
                How many transitions to sample.
        :returns
            obs_batch: np.array
                batch of observations
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_obs_batch: np.array
                next set of observations seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
        """
        if batch_size > 0: # 采集batch_size个样本
            idxes = self.make_index(batch_size)
        else: # 采集缓冲区内所有样本
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    # 采集缓冲区内所有样本
    def collect(self):
        return self.sample(-1)