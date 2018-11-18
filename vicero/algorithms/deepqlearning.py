import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque


class Model(nn.Module):
    # Simple net with one hidden layer
    def __init__(self, input_size, first_layer, second_layer, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(first_layer, second_layer)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(second_layer, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
        
class DQNAgent:
    def __init__(self, env, alpha=.001, epsilon=1.0, gamma=.95, eps_min=.01, eps_decay=.99, memory_length=2000, state_to_reward=None, render=True):

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
        
        # the following 7 lines should be elegantly generalized
        torch.set_default_tensor_type('torch.DoubleTensor')
        device = torch.device('cpu')
        feature_size, action_size = env.observation_space.shape[0], env.action_space.n
        first_layer = 24
        second_layer = 24
        optimizer = torch.optim.Adam
        loss_fct = nn.MSELoss
        
        self.model = Model(feature_size, first_layer, second_layer, action_size)    
        self.memory = deque(maxlen=memory_length)
        self.model = self.model.to(device)
        self.device = device
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

    def winning_action(self, state):
        outputs = self.model(state)
        return outputs.max(0)[1].numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        torch.save(self.model.state_dict(), name)