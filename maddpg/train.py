import sys
sys.path.append('..')
from uav_env import UavEnvironment
from maddpg import MADDPG
import numpy as np
import torch
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

    # 配置网络和缓冲区的初始化参数
    n_agents = 3
    n_states = env.obsvervation_space.dim
    n_actions = env.action_space.dim
    capacity = 100000
    batch_size = 128
    # 循环次数
    n_episode = 10000
    # max_steps = 1000
    episodes_before_train = 100

    win = None
    param = None

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)

    FloatTensor = torch.cuda.FloatTensor if maddpg.use_cuda else torch.FloatTensor

    if load:
        maddpg.load_model('')

    avg_reward = 0.0

    for i_episode in range(n_episode):
        obs = env.reset()
        obs = np.stack(obs)
        obs = obs.astype(float)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        total_reward = 0.0

        isdone = False
        t = 0
        while not isdone:
            obs = obs.type(FloatTensor)
            action = maddpg.select_action(obs).data.cpu()
            # render every 100 episodes to speed up training
            if i_episode % 100 == 0 and t % 10 == 0 and render:
                # cv2.imshow("env", env.render.image)
                filepath = '../img/' + str(i_episode/100) + '-' + str(t) + '.jpg'
                print(filepath)
                cv2.imwrite(filepath, env.render.image)
            obs_, reward, done, _ = env.step(action.numpy())
            reward = torch.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = obs_.astype(float)
            obs_ = torch.from_numpy(obs_).float()
            # 判断一个episode是否结束
            isdone = True
            for d in done:
                if d == 0:
                    isdone = False
                    break
            if isdone is False:
                next_obs = obs_
            else:
                next_obs = None
            total_reward += reward.sum()
            if next_obs is not None:
                maddpg.memory.add(obs.data, action, reward, next_obs.data, done)
            obs = next_obs
            c_loss, a_loss = maddpg.update_policy()
            t += 1

        maddpg.episode_done += 1
        if i_episode % 50 == 0:
            avg_reward /= 50
            print('Episode: %d, reward = %f' % (i_episode, total_reward))
            print('Average reward: %f' % avg_reward)
            avg_reward = 0.0
            maddpg.save_model()
        avg_reward += total_reward

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
            print('MADDPG scale_reward=%f\n' % maddpg.scale_reward + 'agent=%d' % n_agents + '\nlr=0.001\n')

