from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from vicero.policy import Policy


# Q-learning is a RL algorithm based on learning the Q (Quality) function

# TODO:
# - Implement UCB

class Qlearning:

    def __init__(self, env, learning_rate=0.1, learning_rate_decay=lambda lr, i: lr, epsilon=0.1,
                 epsilon_decay=lambda epsilon, i: epsilon, gamma=0.95, table_type="numpy",
                 n_states=None, n_actions=None, discretize=lambda state: state, plotter=None):
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma
        self.discretize = discretize

        self.history = []
        self.plotter = plotter
        self.table_type = table_type

        # Set up Q-table
        if table_type == "numpy":
            self.Q = np.zeros((self.n_states, self.n_actions))
        elif table_type == "dict":
            self.Q = defaultdict(lambda: [0] * n_actions)

    def update_q(self, state, action, reward, next_state):
        # Bellman update
        self.Q[self.discretize(state)][action] += self.learning_rate * (
                reward + self.gamma * np.max(self.Q[self.discretize(next_state)])
                - self.Q[self.discretize(state)][action]
        )

    def exploratory_action(self):
        # Random action
        return np.random.randint(self.n_actions)

    def greedy_action(self, state):
        # Best action found in state
        return np.argmax(self.Q[self.discretize(state)])

    def action(self, state):
        # choose an action
        return self.exploratory_action() if (np.random.random() <= self.epsilon) else self.greedy_action(state)

    def train(self, episodes):
        for i in range(episodes):
            # Reset episode
            total_reward = 0
            done = False
            state = self.env.reset()
            while not done:
                action = self.action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_q(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            self.history.append(total_reward)
            if self.plotter is not None:
                self.plotter(self.history)

            # Decay learning rate and epsilon
            self.epsilon = self.epsilon_decay(self.epsilon, i)
            self.learning_rate = self.learning_rate_decay(self.learning_rate, i)

    def get_epsilon(self):
        return self.epsilon

    def action_distribution(self, state):
        out = torch.tensor(self.Q[self.discretize(state)])
        return nn.Softmax(dim=0)(out)

    def copy_target_policy(self):
        tmp = self.env
        self.env = None
        cpy = deepcopy(self)
        self.env = tmp
        return Policy(cpy.greedy_action)
