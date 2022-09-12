from maddpg import MADDPG
import sys

sys.path.append('..')
from uav_env2 import UavEnvironment
import numpy as np
import torch as th
import cv2
import time

if __name__ == "__main__":
    # 是否载入模型
    load = False
    # 载入环境
    pois = np.load("../data/pois.npy", allow_pickle=True)
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
    batch_size = 256

    n_episode = 5000
    max_steps = 1/(env.uavs[0].cal_energy_loss([]))
    print('max steps per episode:', max_steps)
    episodes_before_train = 100

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    avg_reward = 0.0
    for i_episode in range(n_episode):
        obs = env.reset()
        obs = np.stack(obs)
        obs = obs.astype(float)
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs).float()
        total_reward = 0.0
        for t in range(int(max_steps)):
            obs = obs.type(FloatTensor)
            action = maddpg.select_action(obs).data.cpu()
            # render every 100 episodes to speed up training
            if i_episode % 100 == 0 and t % 20 == 0 and env.is_render:
                filepath = '../img/' + str(i_episode / 100) + '-' + str(t) + '.jpg'
                print(filepath)
                cv2.imwrite(filepath, env.render.image)
            obs_, reward, done, _ = env.step(action.numpy())
            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = obs_.astype(float)
            obs_ = th.from_numpy(obs_).float()
            if t != max_steps - 1:
                next_obs = obs_
            else:
                next_obs = None
            total_reward += reward.sum()
            maddpg.memory.push(obs.data, action, next_obs, reward)
            obs = next_obs
            c_loss, a_loss = maddpg.update_policy()

        maddpg.episode_done += 1
        avg_reward += total_reward
        print('Episode: %d, reward = %f avg_reward = %f' % (i_episode, total_reward, avg_reward/(i_episode + 1)))
        if i_episode % 100 == 0:
            maddpg.save_model()

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
