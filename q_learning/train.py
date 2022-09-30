from __future__ import division
import cv2
import numpy as np
import sys
from qlearning import QLearningTable

sys.path.append('..')
from uav_env import UavEnvironment
from compare import greedy

MAX_EPISODES = 50
MAX_STEPS = 200
MAX_BUFFER = 1000000

pois = np.load("../data/pois.npy", allow_pickle=True)
obstacles = []
n_agents = 2
env = UavEnvironment(pois, obstacles, n_agents)
env.is_render = True
env.share_reward = True

# action_space = [(10, 0), (0, 0), (0, 10), (-10, 0), (0, -10)]
action_space = [(10, 0), (0, 0), (0, 10)]

new_action_space = []
for i in action_space:
    for j in action_space:
        new_action_space.append((i, j))
print(len(new_action_space))
action_space = new_action_space
table = QLearningTable(action_space, e_greedy=0.2)


avg_reward = 0.0
for _ep in range(MAX_EPISODES):
    observation = env.reset()
    observation = tuple(map(tuple, observation))
    total_reward = 0.0
    for r in range(MAX_STEPS):
        action = table.choose_action(observation, explore=True)
        next_obs, reward, done, _ = env.step(action)
        next_obs = tuple(map(tuple, next_obs))
        gameover = 1
        for d in done:
            if d == 0:
                gameover = 0
        table.learn(observation, action, reward[0], next_obs, gameover)
        observation = next_obs
        total_reward += reward[0]
        if gameover == 1:
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
    print(action)
    next_obs, reward, done, _ = env.step(action)
    next_obs = tuple(map(tuple, next_obs))
    observation = next_obs
    total_reward += reward[0]
    print(reward[0])

    gameover = 1
    for d in done:
        if d == 0:
            gameover = 0
    if gameover == 1:
        print(total_reward)
        cv2.imshow("env", env.render.image)
        cv2.waitKey(0)

