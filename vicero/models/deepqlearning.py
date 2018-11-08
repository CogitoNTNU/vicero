import numpy as np
from collections import namedtuple
import random

class DeepQlearning:
    def __init__(self, n_states, n_actions, alpha=0.1, epsilon=0.1, gamma=0.5):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        self.n_states = n_states
        self.n_actions = n_actions

        self.Q = np.zeros((self.n_states, self.n_actions))

    def update_q(self, state_old, action, reward, state_new):
        self.Q[state_old][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def exploratory_action(self, state):
        return np.random.randint(self.n_actions) if (
                    np.random.random() <= self.epsilon) else np.argmax(self.Q[state])

    def winning_action(self, state):
        return np.argmax(self.Q[state])

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
