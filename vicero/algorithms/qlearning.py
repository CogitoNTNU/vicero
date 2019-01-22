import numpy as np
import math
from copy import deepcopy
from vicero.policy import Policy

# Q-learning is a RL algorithm based on learning the Q (Quality) function
# This implementation is the classical tabular one.

# TODO:
# - Implement UCB
# - Cleanup

class Qlearning:
    def __init__(self, env, n_states, n_actions, alpha=0.1, epsilon=0.1, gamma=0.5, shakeup=0.1, discretize=lambda state : state):
        self.env = env
        
        # alpha: learning rate
        self.alpha = alpha
        # epsilon: exploration rate
        self.epsilon = epsilon
        # gamma: future discounted reward rate
        self.gamma = gamma
        # shakeup: (in the lack of a better name) chance of total randomization
        self.shakeup = shakeup
        # discretize: takes a state of continous values and converts them to discrete values
        self.discretize = discretize

        # TODO: fetch these values from env (revamp env wrapper)
        self.n_states = n_states
        self.n_actions = n_actions
        
        # initialize the Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))

    # Implementation of the Bellman equation
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