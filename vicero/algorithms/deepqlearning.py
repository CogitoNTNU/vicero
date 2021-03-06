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

class DQN:
    def __init__(self, env, spec=None, alpha=1e-3, gamma=.95, epsilon_start=0.9, epsilon_end=1e-3, memory_length=2000, state_to_reward=None, render=True, qnet_path=None, qnet=None, plotter=None, caching_interval=1000, plot_durations=False):
        self.plot_durations = plot_durations
        # learning rate
        self.alpha = alpha

        # discount factor
        self.gamma = gamma

        # exploration rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = self.epsilon_start

        self.env = env
        self.state_to_reward = state_to_reward
        self.plotter = plotter
        self.device = torch.device('cpu')

        # the following 4 lines should be elegantly generalized
        torch.set_default_tensor_type('torch.DoubleTensor')
        optimizer = torch.optim.Adam
        loss_fct = nn.MSELoss
        self.n_actions = env.action_space.n
        if qnet is not None:
            self.qnet = qnet
        elif spec is not None:
            self.feature_size = env.observation_space.shape[0]
            self.qnet = NeuralNetwork(self.feature_size, self.n_actions, spec).to(self.device)
        else:
            raise Exception('The qnet, qnet_path and spec argument cannot all be None!')

        if qnet_path is not None:
            self.qnet.load_state_dict(torch.load(qnet_path))

        self.memory = deque(maxlen=memory_length)
        self.optimizer = optimizer(self.qnet.parameters(), lr=self.alpha)
        self.criterion = loss_fct()
        self.render = render

        self.state_visits = {} 
        self.history = []
        self.maxq_history = []
        self.maxq_temp = float('-inf')
        self.loss_history = []
        self.loss_temp = 0
        self.loss_count = 0
        self.cached_qnet = deepcopy(self.qnet)
        self.caching_interval = caching_interval
        self.total_iterations = 0

    def train_episode(self, e, num_episodes, batch_size, training_iter=5000, completion_reward=None, verbose=False, plot=False, eps_decay=True):
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (e / num_episodes)
        
        state = self.env.reset()
        state = torch.from_numpy(state).to(self.device)#torch.from_numpy(np.flip(state,axis=0).copy())
        
        done = False
        score = 0

        progress = 0

        self.maxq_temp = float('-inf')

        duration = 0
        for time in range(training_iter):
            duration = time
            if self.render: self.env.render()

            if state in self.state_visits.keys():
                self.state_visits[state] += 1
            else:
                self.state_visits[state] = 1

            action = self.exploratory_action(state, record_maxq=True)
            next_state, reward, done, _ = self.env.step(action)
            self.total_iterations += 1

            if self.state_to_reward:
                reward = self.state_to_reward(next_state)
                
            if completion_reward is not None and done:
                reward = completion_reward
                
            score += reward

            next_state = torch.from_numpy(next_state).to(self.device)#np.flip(next_state,axis=0).copy()).to(self.device)

            self.remember(state, action, reward, next_state, done)
                
            state = next_state
                
            if done: break

            if self.total_iterations % self.caching_interval == 0:
                self.cached_qnet = deepcopy(self.qnet)

            if len(self.memory) > batch_size:
                self.replay(batch_size, eps_decay)

        self.history.append(duration if self.plot_durations else score)
        self.maxq_history.append(self.maxq_temp)
            
        if self.plotter is not None:
            self.plotter(self.history)
            
        if verbose:
            print("episode: {}/{}, score: {:.2}, e: {:.2}, maxQ={:.2}, len(memory)={}".format(e, num_episodes, float(score), self.epsilon, self.maxq_temp, len(self.memory)))
            if self.loss_count > 0:
                self.loss_history.append(self.loss_temp / self.loss_count)
                self.loss_temp = 0
                self.loss_count = 0
                
    def train(self, num_episodes, batch_size, training_iter=5000, completion_reward=None, verbose=False, plot=False, eps_decay=True):
        for e in range(num_episodes):
            self.train_episode(e, num_episodes, batch_size, training_iter, completion_reward, verbose, plot, eps_decay)

            
    def replay(self, batch_size, eps_decay):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = state.to(self.device)
            reward = torch.tensor(reward, dtype=torch.double, requires_grad=False)
            
            target = reward
            
            outputs = self.cached_qnet(next_state)
            target = (reward + self.gamma * torch.max(outputs))

            target_f = self.qnet(state)
            target_f[action] = target
            
            prediction = self.qnet(state)
            
            loss = self.criterion(prediction, target_f)
            self.loss_temp += float(loss)
            self.loss_count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def exploratory_action(self, state, record_maxq=False):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.n_actions))
        
        outputs = self.qnet(state)
        
        if record_maxq:
            self.maxq_temp = max([self.maxq_temp] + list(outputs))
        return outputs.max(0)[1].numpy()

    def greedy_action(self, state):
        outputs = self.qnet(state)
        return outputs.max(0)[1].numpy()

    def action_distribution(self, state):
        state = torch.from_numpy(state).to(self.device)
        out = self.qnet(state)
        return nn.Softmax(dim=0)(out)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        torch.save(self.qnet.state_dict(), name)

    def copy_target_policy(self, verbose=False):
        cpy = deepcopy(self.qnet)
        device = self.device
        def policy(state):
            state = torch.from_numpy(state).to(device)
            #state = torch.from_numpy(np.flip(state,axis=0).copy())
            #state = state.to(device)
            distribution = cpy(state)
            if verbose:
                print('state:', state) 
                print('Q(state):', distribution)
            return distribution.max(0)[1].numpy()

        return Policy(policy)