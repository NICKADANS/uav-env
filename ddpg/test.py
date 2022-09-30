from __future__ import division

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import sys

sys.path.append('..')
from uav_env import UavEnvironment
import ddpg
import buffer
import utils

MAX_EPISODES = 1
MAX_STEPS = 200
MAX_BUFFER = 1000000

pois = np.load("../data/pois.npy", allow_pickle=True)
obstacles = []
n_agents = 1
env = UavEnvironment(pois, obstacles, n_agents)
env.is_render = True
env.share_reward = True

S_DIM = env.obsvervation_space.dim * len(env.uavs)
A_DIM = env.action_space.dim * len(env.uavs)
A_MAX = 10.0

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = ddpg.Trainer(S_DIM, A_DIM, A_MAX, ram)
trainer.load_models(20)
avg_reward = 0.0

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    observation = np.float32(np.array(observation, dtype=object).flatten())
    total_reward = 0.0
    for r in range(MAX_STEPS):
        observation = np.float32(np.array(observation, dtype=object).flatten())
        action = trainer.get_exploitation_action(observation)
        action = utils.trans_env_action(action)
        act = np.array(action).reshape((len(action) // env.action_space.dim, env.action_space.dim))
        new_observation, reward, done, _ = env.step(act)
        new_observation = np.float32(np.array(new_observation, dtype=object).flatten())
        reward = reward[0]
        total_reward += reward
        observation = new_observation
        # 判断游戏是否结束
        gameover = True
        for d in done:
            if d == 0:
                gameover = False
        action = utils.trans_net_action(action)
        if _ep % 20 == 0 and r % 20 == 0 and env.is_render:
            filepath = '../img/' + str(_ep / 100) + '-' + str(r) + '.jpg'
            print(filepath)
            cv2.imwrite(filepath, env.render.image)

        if gameover:
            break
    avg_reward += total_reward
    print('Episode: %d, reward = %f, avg_reward = %f' % (_ep, total_reward, avg_reward / (_ep + 1)))
    # check memory consumption and clear memory
    gc.collect()
