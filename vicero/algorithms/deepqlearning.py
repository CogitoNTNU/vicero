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
    def __init__(self, env, spec, alpha=.001, gamma=.95, epsilon_start=1.0, epsilon_end=1.0, memory_length=2000, state_to_reward=None, render=True, anet_path=None):

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
        
        self.device = torch.device('cpu')

        # the following 4 lines should be elegantly generalized
        torch.set_default_tensor_type('torch.DoubleTensor')
        feature_size, self.n_actions = env.observation_space.shape[0], env.action_space.n
        optimizer = torch.optim.Adam
        loss_fct = nn.MSELoss
        
        self.nnet = NeuralNetwork(feature_size, self.n_actions, spec).to(self.device)
        if anet_path is not None:
            self.nnet.load_state_dict(torch.load(anet_path))

        self.memory = deque(maxlen=memory_length)
        self.optimizer = optimizer(self.nnet.parameters(), lr=self.alpha)
        self.criterion = loss_fct()
        self.render = render
        
        self.history = []
        self.maxq_history = []
        self.maxq_temp = float('-inf')
        self.loss_history = []
        self.loss_temp = 0
        self.loss_count = 0

    def train(self, num_episodes, batch_size, training_iter=500, completion_reward=None, verbose=False, plot=False, eps_decay=True):
        
        for e in range(num_episodes):
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (e / num_episodes)
            state = self.env.reset()
            state = torch.from_numpy(state).to(self.device)#torch.from_numpy(np.flip(state,axis=0).copy())
        
            done = False
            score = 0

            progress = 0

            self.maxq_temp = float('-inf')
            for time in range(training_iter):
                if self.render:
                    self.env.render()
                        
                action = self.exploratory_action(state, record_maxq=True)
                next_state, reward, done, _ = self.env.step(action)
                
                if self.state_to_reward:
                    reward = self.state_to_reward(next_state)
                
                if completion_reward is not None and done:
                    reward = completion_reward
                
                score += reward

                next_state = torch.from_numpy(next_state).to(self.device)#np.flip(next_state,axis=0).copy()).to(self.device)

                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                
                if done: break
                
                if len(self.memory) > batch_size:
                    self.replay(batch_size, eps_decay)

            if verbose:
                print("episode: {}/{}, score: {}, e: {:.2}, maxQ={:.2}".format(e, num_episodes, score, self.epsilon, self.maxq_temp))
                self.history.append(score)
                self.maxq_history.append(self.maxq_temp)
                if self.loss_count > 0:
                    self.loss_history.append(self.loss_temp / self.loss_count)
                    self.loss_temp = 0
                    self.loss_count = 0
                

    def replay(self, batch_size, eps_decay):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = state.to(self.device)
            reward = torch.tensor(reward, dtype=torch.double, requires_grad=False)
            target = reward
            if not done:
                outputs = self.nnet(next_state)
                target = (reward + self.gamma * torch.max(outputs))

            target_f = self.nnet(state)
            target_f[action] = target
            prediction = self.nnet(state)
            #print(prediction)

            loss = self.criterion(prediction, target_f)
            self.loss_temp += float(loss)
            self.loss_count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def exploratory_action(self, state, record_maxq=False):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(self.n_actions))
        outputs = self.nnet(state)
        
        if record_maxq:
            self.maxq_temp = max([self.maxq_temp] + list(outputs))

        return outputs.max(0)[1].numpy()

    def greedy_action(self, state):
        outputs = self.nnet(state)
        return outputs.max(0)[1].numpy()

    def remember(self, state, action, reward, next_state, done):
        #if reward == 0 and np.random.uniform() < 0.95:
        #    return # test
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        torch.save(self.nnet.state_dict(), name)

    def copy_target_policy(self, verbose=False):
        cpy = deepcopy(self.nnet)
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