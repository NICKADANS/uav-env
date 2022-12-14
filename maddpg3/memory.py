# 经验缓冲区
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
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        return obses_t, actions, rewards, obses_tp1, dones

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
        if batch_size > 0:  # 采集batch_size个样本
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        else:  # 采集缓冲区内所有样本
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    # 采集缓冲区内所有样本
    def collect(self):
        return self.sample(-1)
