import torch as T
import torch.nn.functional as F
from agent import Agent


class MADDPG:
    def __init__(self, obs_dim, act_dim, n_agents, batch_size, episode_before_train):
        self.agents = []
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.gamma = 0.95
        self.decay = 0.99998
        self.episode_before_train = episode_before_train
        self.episode_done = 0
        for _ in range(self.n_agents):
            self.agents.append(Agent(obs_dim, act_dim, n_agents))

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def choose_action(self, obs):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(obs[i])
            if agent.epsilon > 0.01:
                agent.epsilon *= self.decay
            actions.append(action)
        return actions

    def learn(self, memory):
        if self.episode_done < self.episode_before_train:
            return
        device = self.agents[0].actor.device

        # 从batch中采样
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memory.sample(
            batch_size=self.batch_size)
        states = T.tensor(batch_states, dtype=T.float).to(device)
        actions = T.tensor(batch_actions, dtype=T.float).to(device)
        rewards = T.tensor(batch_rewards, dtype=T.float).to(device)
        next_states = T.tensor(batch_next_states, dtype=T.float).to(device)
        dones = T.tensor(batch_dones).to(device)

        # 计算下一个行为
        next_actions = []
        for i, agent in enumerate(self.agents):
            next_actions.append(agent.actor_target(next_states[:, i, :]))
        next_actions = T.stack(next_actions).to(device)
        next_actions = next_actions.transpose(0, 1).contiguous()

        for i, agent in enumerate(self.agents):
            # 更新 critic 网络
            q_current = agent.critic.forward(states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
            q_next = agent.critic_target.forward(next_states.view(self.batch_size, -1), next_actions.view(self.batch_size, -1))
            for done in dones:
                if done[i] == 1:
                    q_next[i] = 0.0
            q_target = rewards[:, i].view(self.batch_size, -1) + self.gamma * q_next
            critic_loss = F.mse_loss(q_current, q_target)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # 更新 actor 网络
            state_i = states[:, i, :]
            action_i = agent.actor.forward(state_i)
            # 重新选择联合动作中当前agent的动作，其他agent的动作不变
            actions_clone = actions.clone()
            actions_clone[:, i, :] = action_i
            actor_loss = agent.critic.forward(states.view(self.batch_size, -1), actions_clone.view(self.batch_size, -1)).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            # 更新网络
            agent.soft_update()
