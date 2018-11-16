import numpy as np
import math
from copy import deepcopy
from vicero.policy import Policy

class Qlearning:
    def __init__(self, env, n_states, n_actions, alpha=0.1, epsilon=0.1, gamma=0.5, shakeup=0.1, discretize=lambda state : state):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.shakeup = shakeup
        self.discretize = discretize

        self.n_states = n_states
        self.n_actions = n_actions
        
        self.Q = np.zeros((self.n_states, self.n_actions))

    def update_q(self, state_old, action, reward, state_new):
        self.Q[self.discretize(state_old)][action] += self.alpha * (reward + self.gamma * np.max(self.Q[self.discretize(state_new)]) - self.Q[self.discretize(state_old)][action])
        
    def exploratory_action(self, state):
        return np.random.randint(self.n_actions) if (np.random.random() <= self.epsilon) else self.greedy_action(state)

    def greedy_action(self, state):
        return np.argmax(self.Q[self.discretize(state)])

    def train(self, iterations):
        for _ in range(iterations):
            if np.random.random() <= self.shakeup:
                self.env.randomize()
            
            state_old = self.env.state
            action = self.exploratory_action(self.env.state)
            state, reward, done, board = self.env.step(action)
            self.update_q(state_old, action, reward, state)
            
            if done:
                self.env.reset()
    
    def get_epsilon(self):
        return self.epsilon

    def copy_target_policy(self):
        cpy = deepcopy(self)
        return Policy(cpy.greedy_action)