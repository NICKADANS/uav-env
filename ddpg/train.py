from __future__ import division
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc
import sys
sys.path.append('..')
from uav_env2 import UavEnvironment
import ddpg
import buffer

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000

pois = np.load("../data/pois.npy", allow_pickle=True)
obstacles = []
n_agents = 3
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

for _ep in range(MAX_EPISODES):
	observation = env.reset()


	print('EPISODE :- ', _ep)
	for r in range(MAX_STEPS):
		observation = np.float32(np.array(observation, dtype=object).flatten())
		action = trainer.get_exploration_action(observation)
		act = np.array(action).reshape((len(action)//env.action_space.dim, env.action_space.dim))
		# if _ep%5 == 0:
		# 	# validate every 5th episode
		# 	action = trainer.get_exploitation_action(state)
		# else:
		# 	# get action based on observation, use exploration policy here
		# 	action = trainer.get_exploration_action(state)
		new_observation, reward, done, _ = env.step(act)
		new_observation = np.float32(np.array(new_observation, dtype=object).flatten())
		reward = reward[0]
		observation = new_observation
		# 判断游戏是否结束
		gameover = True
		for d in done:
			if d == 0:
				gameover = False
		if not gameover:
			# push this exp in ram
			ram.add(observation, action, reward, new_observation)
		# perform optimization
		trainer.optimize()

		if gameover:
			break

	# check memory consumption and clear memory
	gc.collect()

	if _ep % 100 == 0:
		trainer.save_models(_ep)