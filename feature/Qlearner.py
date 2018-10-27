import numpy as np
import tensorflow  as tf
import math


class Qlearner:
    def __init__(self, num_actions=None, num_states=None,  gamma=1.0,
                 min_alpha=0.1, min_epsilon=0.1, env=None, win_criterion=None):
        # Assumes all states have same number of possible actions in them
        # Possible further improvements are implementing other optimization rules (Adam, Adagrad,..)

        self.alpha = min_alpha  # learning rate
        self.epsilon = min_epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.num_actions = num_actions
        self.num_states = num_states
        self.win_criterion = win_criterion
        self.env = env
        self.Q = np.zeros((self.num_states, self.num_actions))

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
                self.Q[state])

    def get_epsilon(self):
        return self.epsilon

    def get_alpha(self):
        return self.alpha
