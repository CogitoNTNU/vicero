import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
from vicero.policy import Policy
from copy import deepcopy
from vicero.algorithms.common.neuralnetwork import NeuralNetwork, NetworkSpecification

# DQN (Deep Q Networks)
# DQN is an approximated variant of Q-learning
# Significant differences:
# - The table is replaced by a neural network
# - Experiences are stored in a replay buffer, which is used to train the network
# The update rule is similarly to Q-learning based on the Bellman equation

# In vicero, certain shared traits of different algorithms are pulled out
# and placed in the common module. Neural networks and replay buffers are
# among those. This allows for this module to be written more cleanly, with
# a more pure focus on the reinforcement learning.

class DQNAgent:
    def __init__(self, env, spec, alpha=.001, epsilon=1.0, gamma=.95, eps_min=.01, eps_decay=.99, memory_length=2000, state_to_reward=None, render=True):

        # learning rate
        self.alpha = alpha

        # discount factor
        self.gamma = gamma

        # exploration rate
        self.epsilon = epsilon
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        
        self.env = env
        self.state_to_reward = state_to_reward
        
        # the following 5 lines should be elegantly generalized
        torch.set_default_tensor_type('torch.DoubleTensor')
        device = torch.device('cpu')
        feature_size, action_size = env.observation_space.shape[0], env.action_space.n
        optimizer = torch.optim.Adam
        loss_fct = nn.MSELoss
        
        self.model = NeuralNetwork(feature_size, action_size, spec)    
        self.memory = deque(maxlen=memory_length)
        self.model = self.model.to(device)
        self.device = device
        print(self.model.parameters())
        self.optimizer = optimizer(self.model.parameters(), lr=self.alpha)
        self.criterion = loss_fct()
        
        self.render = render
        self.n_actions = action_size

    def train(self, num_episodes, batch_size, training_iter=500, completion_reward=0, verbose=False,
              plot=False, eps_decay=True):
        # batch_size : number of replays to perform at each training step

        for e in range(num_episodes):
            state = self.env.reset()
            #state = torch.from_numpy(state)
            state = torch.from_numpy(np.flip(state,axis=0).copy())
            state = state.to(self.device)
            for time in range(training_iter):
                if self.render:
                    self.env.render()

                action = self.exploratory_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if self.state_to_reward:
                    reward = self.state_to_reward(next_state)
                
                reward = reward if not done else completion_reward

                #next_state = torch.from_numpy(next_state)
                next_state = torch.from_numpy(np.flip(next_state,axis=0).copy())

                next_state = next_state.to(self.device)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done and verbose:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, num_episodes, time, self.epsilon))
                    break

                if len(self.memory) > batch_size:
                    self.replay(batch_size, eps_decay)

    def replay(self, batch_size, eps_decay):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = state.to(self.device)
            reward = torch.tensor(reward, dtype=torch.double, requires_grad=False)
            target = reward
            if not done:
                outputs = self.model(next_state)
                target = (reward + self.gamma * torch.max(outputs))

            target_f = self.model(state)
            target_f[action] = target
            prediction = self.model(state)

            loss = self.criterion(prediction, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if (self.epsilon > self.epsilon_min) and eps_decay:
            self.epsilon *= self.epsilon_decay

    def exploratory_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.n_actions))
        outputs = self.model(state)
        return outputs.max(0)[1].numpy()

    def greedy_action(self, state):
        outputs = self.model(state)
        return outputs.max(0)[1].numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def copy_target_policy(self):
        cpy = deepcopy(self.model)
        device = self.device
        def policy(state):
            state = torch.from_numpy(np.flip(state,axis=0).copy())
            state = state.to(device)
            return cpy(state).max(0)[1].numpy()

        return Policy(policy)