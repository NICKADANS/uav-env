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
from compare import greedy

MAX_EPISODES = 5000
MAX_STEPS = 200
MAX_BUFFER = 100000

pois = np.load("../data/pois.npy", allow_pickle=True)
obstacles = []
n_agents = 1
env = UavEnvironment(pois, obstacles, n_agents)
env.is_render = True
env.share_reward = True

S_DIM = (env.obsvervation_space.dim[0], env.obsvervation_space.dim[1]*n_agents, env.obsvervation_space.dim[2])
A_DIM = env.action_space.dim * n_agents
A_MAX = env.uavs[0].v_max

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = ddpg.Trainer(S_DIM, A_DIM, A_MAX, ram)

avg_reward = 0.0
for _ep in range(MAX_EPISODES):
	observation = env.reset()
	observation = utils.concat_obs(observation)
	total_reward = 0.0
	for r in range(MAX_STEPS):
		action = trainer.get_exploration_action([observation])
		action = utils.trans_env_action(action[0], A_MAX)


		act = np.array(action).reshape((len(action) // env.action_space.dim, env.action_space.dim))
		# cv2.imshow("s", np.transpose(observation, (1, 2, 0)))
		# cv2.waitKey(0)
		new_observation, reward, done, _ = env.step(act)
		# print(env.uavs[0].x, env.uavs[0].y, new_observation[0])
		new_observation = utils.concat_obs(new_observation)
		reward = reward[0]
		total_reward += reward
		# 判断游戏是否结束
		gameover = True
		for d in done:
			if d == 0:
				gameover = False
		action = utils.trans_net_action(action, A_MAX)
		ram.add(observation, action, reward, new_observation)
		# perform optimization
		trainer.optimize()
		observation = new_observation
		if _ep % 20 == 0 and r % 20 == 0 and env.is_render:
			filepath = '../img/' + str(_ep / 100) + '-' + str(r) + '.jpg'
			print(filepath)
			cv2.imwrite(filepath, env.render.image)
		if gameover:
			break
	trainer.var = trainer.var - 0.001 if trainer.var > 0.1 else 0.1
	avg_reward += total_reward
	print('Episode: %d, reward = %f, avg_reward = %f' % (_ep, total_reward, avg_reward/(_ep + 1)))
	# check memory consumption and clear memory
	gc.collect()

	if _ep % 100 == 0:
		trainer.save_models(_ep)