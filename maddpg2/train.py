from maddpg import MADDPG
import sys

sys.path.append('..')
from uav_env import UavEnvironment
import numpy as np
import torch as th
import cv2

if __name__ == "__main__":
    # 是否渲染
    render = False
    # 是否载入模型
    load = False
    # 载入环境
    pois = np.load("../data/pois.npy", allow_pickle=True)
    # obstacles = np.load("../data/obstacles.npy")
    obstacles = []
    env = UavEnvironment(pois, obstacles, 3)

    # 配置参数
    np.random.seed(1234)
    th.manual_seed(1234)
    n_agents = 3
    n_states = env.obsvervation_space.dim
    n_actions = env.action_space.dim
    capacity = 100000
    batch_size = 128

    n_episode = 10000
    max_steps = 1/(env.uavs[0].cal_energy_loss([]))
    print('max steps per episode:', max_steps)
    episodes_before_train = 100

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor

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
            if i_episode % 100 == 0 and t % 10 == 0 and render:
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
        print('Episode: %d, reward = %f' % (i_episode, total_reward))

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
