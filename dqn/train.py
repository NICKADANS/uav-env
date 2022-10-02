#!/usr/bin/env python3
from buffer import ExperienceBuffer
from model import DQN
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from agent import Agent

sys.path.append('..')
from dqn_env import UavEnvironment

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 50000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones).to(device)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    MAX_EPISODES = 5000
    MAX_STEPS = 200
    pois = np.load("../data/pois.npy", allow_pickle=True)
    obstacles = []
    n_agents = 1
    env = UavEnvironment(pois, obstacles, n_agents)
    env.is_render = True
    env.share_reward = True

    S_DIM = (env.obsvervation_space.dim[0], env.obsvervation_space.dim[1], env.obsvervation_space.dim[2])
    A_DIM = env.action_space.n

    print(' State Dimensions :- ', S_DIM)
    print(' Action Dimensions :- ', A_DIM)

    # net
    net = DQN(env.obsvervation_space.dim, env.action_space.n).to(device)
    tgt_net = DQN(env.obsvervation_space.dim, env.action_space.n).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    for ep in range(MAX_EPISODES):
        is_done = 0
        while is_done == 0:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            reward, is_done = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                m_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, reward %.3f, " "eps %.2f" % (frame_idx, len(total_rewards), m_reward, epsilon,))
                if best_m_reward is None or best_m_reward < m_reward:
                    # torch.save(net.state_dict(), "best_%.0f.dat" % m_reward)
                    if best_m_reward is not None:
                        print("Best reward updated %.3f -> %.3f" % (best_m_reward, m_reward))
                    best_m_reward = m_reward

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
