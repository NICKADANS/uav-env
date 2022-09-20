from maddpg import MADDPG
from memory import ReplayBuffer
import sys
sys.path.append('..')
from uav_env2 import UavEnvironment
import numpy as np
import torch as T
import cv2
import time

if __name__ == "__main__":
    # 是否载入模型
    load = False
    # 载入环境
    pois = np.load("../data/easypoi.npy", allow_pickle=True)
    # obstacles = np.load("../data/obstacles.npy")
    obstacles = []
    env = UavEnvironment(pois, obstacles, 3)
    env.is_render = True
    env.share_reward = True
    # 配置参数
    np.random.seed(int(time.time()))
    n_agents = 3
    n_states = env.obsvervation_space.dim
    n_actions = env.action_space.dim
    capacity = 100000
    batch_size = 16
    n_episode = 10000
    episodes_before_train = 10
    max_steps = 1/(env.uavs[0].cal_energy_loss([]))
    print('max steps per episode:', max_steps)
    maddpg = MADDPG(n_states, n_actions, n_agents, batch_size, episodes_before_train)
    memory = ReplayBuffer(capacity)
    avg_reward = 0.0
    for i_episode in range(n_episode):
        obs = env.reset()
        obs = np.stack(obs).astype(float)
        total_reward = 0.0
        for t in range(int(max_steps)):
            action = maddpg.choose_action(obs)
            # render every 100 episodes to speed up training
            if i_episode % 100 == 0 and t % 20 == 0 and env.is_render:
                filepath = '../img/' + str(i_episode / 100) + '-' + str(t) + '.jpg'
                print(filepath)
                cv2.imwrite(filepath, env.render.image)

            obs_, reward, done, _ = env.step(action)
            obs_ = np.stack(obs_).astype(float)
            memory.add(obs, action, reward, obs_, done)
            total_reward += reward[0]
            obs = obs_
            maddpg.learn(memory)

        maddpg.episode_done += 1
        avg_reward += total_reward
        print('Episode: %d, reward = %f avg_reward = %f' % (i_episode, total_reward, avg_reward/(i_episode + 1)))

        if maddpg.episode_done == maddpg.episode_before_train:
            print('training now begins...')
