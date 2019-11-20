import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from itertools import count
import numpy as np

class Reinforce:
    def __init__(self, env, polinet, learning_rate=0.01, gamma=0.99, batch_size=5, plotter=None):
        self.policy_net = polinet
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.episode_history = []
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.plotter = plotter
        self.binary_action_space = (list(polinet.modules())[-1].out_features == 1)

    def gradient_step(self, trajectories):
        
        
        for i in range(len(trajectories)):

            # Discount rewards        
    
            trajectory = trajectories[i]
            discounted_reward = 0
            for j in reversed(range(len(trajectory))):
                state, action, reward = trajectory[j]
                discounted_reward = discounted_reward * self.gamma + reward
                trajectories[i][j] = (state, action, discounted_reward)

            # Normalize rewards

            rewards = [frame[2] for frame in trajectory]

            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)

            for j in range(len(trajectory)):
                state, action, reward = trajectory[j]
                normalized_reward = (reward - reward_mean) / reward_std 
                trajectories[i][j] = (state, action, normalized_reward)

        # Calculate gradients

        self.optimizer.zero_grad()
        for i in range(len(trajectories)):
            trajectory = trajectories[i]
            for j in range(len(trajectory)):
                state, action, reward = trajectory[j]

                probs = self.policy_net(state)
                
                m = Bernoulli(probs) if self.binary_action_space else Categorical(probs)

                loss = -m.log_prob(action) * reward
                
                loss.backward()
            
        self.optimizer.step()

    def train(self, n_episodes):
        render = False
        trajectories = []
        for e in range(n_episodes):
            
            trajectory = []
            state = torch.from_numpy(self.env.reset()).float()
            if render: self.env.render(mode='rgb_array')

            for t in count():
                probs = self.policy_net(state)
                
                if self.binary_action_space:
                    m = Bernoulli(probs)
                    action = m.sample().data.numpy().astype(int)[0]
                else: 
                    m = Categorical(probs)
                    action = m.sample()
                
                next_state, reward, done, _ = self.env.step(action.item())
                
                if render: self.env.render(mode='rgb_array')
                
                # TODO: add log prob to trajectory, optimization

                trajectory.append((state, action, reward))            
                state = torch.from_numpy(next_state).float()
            
                if done:
                    trajectories.append(trajectory)
                    self.episode_history.append(t + 1)
                    if self.plotter: self.plotter(self.episode_history)
                    break

            if e > 0 and e % self.batch_size == 0:
                self.gradient_step(trajectories)
                trajectories = []