import argparse
import time
import numpy as np
import torch
import sys
from dqn import DeepQTable

sys.path.append('..')
from dqn_env import UavEnvironment

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = 1e-4
REPLAY_START_SIZE = 10000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    MAX_EPISODES = 1
    MAX_STEPS = 200
    pois = np.load("../data/pois.npy", allow_pickle=True)
    obstacles = [[650, 650], [300, 400]]
    n_agents = 2
    env = UavEnvironment(pois, obstacles, n_agents, uav_init_pos=[])
    env.is_render = True
    env.share_reward = True

    S_DIM = (env.obsvervation_space.dim[0], env.obsvervation_space.dim[1], env.obsvervation_space.dim[2])
    A_DIM = env.action_space.n

    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)

    # dqn
    dqn = DeepQTable(env, env.obsvervation_space.dim, env.action_space.n, n_agents, device)
    dqn.load_models(508)

    for ep in range(MAX_EPISODES):
        gameover = False
        while gameover is False:
            reward, gameover = dqn.play_step(epsilon=0, render=True)


