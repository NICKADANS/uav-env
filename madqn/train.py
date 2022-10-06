import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from dqn import DeepQTable

sys.path.append('..')
from dqn_env import UavEnvironment

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 200000
EPSILON_START = 1.0
EPSILON_FINAL = 0.05

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    MAX_EPISODES = 5000
    MAX_STEPS = 200
    pois = np.load("../data/pois.npy", allow_pickle=True)
    obstacles = []
    n_agents = 2
    uav_init_pos = [[250, 250], [750, 750]]
    env = UavEnvironment(pois, obstacles, n_agents, uav_init_pos)
    env.is_render = True
    env.share_reward = True

    S_DIM = (env.obsvervation_space.dim[0], env.obsvervation_space.dim[1], env.obsvervation_space.dim[2])
    A_DIM = env.action_space.n

    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)

    # dqn
    dqn = DeepQTable(env, env.obsvervation_space.dim, env.action_space.n, n_agents, device)
    epsilon = EPSILON_START
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    for ep in range(MAX_EPISODES):
        gameover = False
        while gameover is False:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward, gameover = dqn.play_step(epsilon)
            if reward is not None:
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                print(
                    "%d: done %d games, reward %.3f, " "eps %.2f" % (frame_idx, len(total_rewards), m_reward, epsilon,))
                if best_m_reward is None or int(best_m_reward + 2) < int(m_reward):
                    dqn.save_models(int(m_reward))
                    if best_m_reward is not None:
                        print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                    best_m_reward = m_reward

            if len(dqn.buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                dqn.hard_update()

            dqn.optimize()
