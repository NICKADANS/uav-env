from hybrid import MADDPG
import sys

sys.path.append('..')
from uav_env2 import UavEnvironment
import numpy as np
import torch as th
import cv2
import time
from compare import global_greedy

if __name__ == "__main__":
    # 是否载入模型
    load = False
    # 载入环境
    pois = np.load("../data/pois.npy", allow_pickle=True)
    # obstacles = np.load("../data/obstacles.npy")
    obstacles = []
    env = UavEnvironment(pois, obstacles, 1)
    env.is_render = True
    env.share_reward = True
    # 配置参数
    np.random.seed(int(time.time()))
    n_agents = 1
    n_states = env.obsvervation_space.dim
    n_actions = env.action_space.dim
    capacity = 100000
    batch_size = 256
    n_episode = 5000
    episodes_before_train = 10
    max_steps = 1/(env.uavs[0].cal_energy_loss([]))
    print('max steps per episode:', max_steps)

    maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity, episodes_before_train)
    maddpg.load_model()
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    avg_reward = 0.0
    for i_episode in range(n_episode):
        obs = env.reset()
        obs = np.stack(obs).astype(float)
        obs = th.FloatTensor(obs).type(FloatTensor)
        total_reward = 0.0
        for t in range(int(max_steps)):
            action = maddpg.select_action(obs, exploration=False).data.cpu()
            # render every 100 episodes to speed up training
            if i_episode % 100 == 0 and t % 20 == 0 and env.is_render:
                filepath = '../img/' + str(i_episode / 100) + '-' + str(t) + '.jpg'
                print(filepath)
                cv2.imwrite(filepath, env.render.image)
            obs_, reward, done, _ = env.step(action.numpy())
            obs_ = np.stack(obs_).astype(float)
            obs_ = th.FloatTensor(obs_).type(FloatTensor)
            done = th.FloatTensor(done).type(FloatTensor)
            reward = th.FloatTensor(reward).type(FloatTensor)
            total_reward += reward[0]
            obs = obs_
        if i_episode > episodes_before_train and maddpg.epsilon > 0.05:
            maddpg.epsilon -= 0.001

        maddpg.episode_done += 1
        avg_reward += total_reward
        print('Episode: %d, reward = %f avg_reward = %f' % (i_episode, total_reward, avg_reward/(i_episode + 1)))

