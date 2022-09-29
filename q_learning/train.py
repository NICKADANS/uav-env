from __future__ import division
import cv2
import numpy as np
import sys
from qlearning import QLearningTable

sys.path.append('..')
from uav_env import UavEnvironment
from compare import greedy

MAX_EPISODES = 200
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

action_space = [(10, 0), (0, 0), (0, 10), (-10, 0), (0, -10)]
table = QLearningTable(action_space, e_greedy=0.2)


avg_reward = 0.0
for _ep in range(MAX_EPISODES):
    observation = env.reset()
    observation = tuple(map(tuple, observation))
    total_reward = 0.0
    for r in range(MAX_STEPS):
        action = table.choose_action(observation, explore=True)
        next_obs, reward, done, _ = env.step([action])
        next_obs = tuple(map(tuple, next_obs))
        table.learn(observation, action, reward[0], next_obs, done[0])
        observation = next_obs
        total_reward += reward[0]
        if done[0] == 1:
            break
    table.epsilon = table.epsilon + 0.01 if table.epsilon < 0.90 else 0.90
    avg_reward += total_reward
    print('Episode: %d, reward = %f, avg_reward = %f' % (_ep, total_reward, avg_reward / (_ep + 1)))

gameover = False
observation = env.reset()
observation = tuple(map(tuple, observation))
total_reward = 0.0
while not gameover:
    action = table.choose_action(observation, explore=False)
    print(table.q_table[observation, (10, 0)], table.q_table[observation, (0, 0)], table.q_table[observation, (0, 10)], action)
    next_obs, reward, done, _ = env.step([action])
    next_obs = tuple(map(tuple, next_obs))
    table.learn(observation, action, reward[0], next_obs, done[0])

    observation = next_obs
    total_reward += reward[0]
    print(reward[0])
    if done[0] == 1:
        gameover = True
        print(total_reward)
        cv2.imshow("env", env.render.image)
        cv2.waitKey(0)

