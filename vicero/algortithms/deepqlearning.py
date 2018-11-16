import numpy as np
import torch
import random
from collections import deque

class DQNAgent:
    def __init__(self, model, env, n_states, n_actions, device, optimizer = None, loss_fct = None,
                 lr=.001, epsilon=1.0, gamma=.95, eps_min=.01, eps_decay=.995, memory_length=2000):
        self.learning_rate = lr
        self.epsilon = epsilon
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.memory = deque(maxlen=memory_length)
        self.model = model.to(device)
        self.env = env
        self.device = device
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.criterion = loss_fct()

        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions

    def train(self, num_episodes, batch_size, training_iter=500, completion_reward=0, verbose=False,
              plot=False, eps_decay=True):
        # batch_size : number of replays to perform at each training step


        for e in range(num_episodes):
            state = self.env.reset()
            state = torch.from_numpy(state)
            state = state.to(self.device)
            for time in range(training_iter):
                if plot:
                    self.env.render()

                action = self.exploratory_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if not done else completion_reward
                next_state = torch.from_numpy(next_state)

                next_state = next_state.to(self.device)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done and verbose:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, num_episodes, time, self.epsilon))
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
                target = (reward + self.gamma *
                          torch.max(outputs))

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
            return random.randrange(self.n_actions)
        outputs = self.model(state)
        return outputs.max(0)[1].numpy()


    def winning_action(self, state):
        outputs = self.model(state)
        return outputs.max(0)[1].numpy()

    def get_epsilon(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

