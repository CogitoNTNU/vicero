import numpy as np
import tensorflow  as tf
import math


class Qlearner:
    def __init__(self, params, num_actions=None, num_states=None, env=None):
        # Assumes all states have same number of possible actions in them
        # Possible further improvements are implementing other optimization rules (Adam, Adagrad,..)

        self.alpha = params.get('alpha', 0.1)  # learning rate
        self.epsilon = params.get('epsilon', 0.1)  # exploration rate
        self.gamma = params.get('gamma', 0.8)  # discount factor

        self.num_actions = num_actions
        self.num_states = num_states
        #self.env = env
        self.Q = np.zeros((self.num_states, self.num_actions))

    def update_q(self, state_old, action, reward, state_new):
        self.Q[state_old][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def exploratory_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        return np.random.randint(self.num_actions) if (np.random.random() <= epsilon) else np.argmax(
                self.Q[state])

    def winning_action(self, state):
        return np.argmax(self.Q[state])

    def get_epsilon(self):
        return self.epsilon

    def get_alpha(self):
        return self.alpha

    def explore(self, num_iter, discretize_fct=None):
        # num_iter gives the number of times the method will try to solve the environment

        for i in range(num_iter):
            # alpha is the learning rate and is used to update the values in Q
            # A larger epsilon means more random actions in the exploration step (0 < epsilon <= 1)

            previous_state = discretize_fct(self.env.reset())
            alpha = self.get_alpha()
            epsilon = self.get_epsilon()
            done = False
            i = 0

            while not done:
                action = self.exploratory_action(previous_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = discretize_fct(obs)
                self.update_q(previous_state, action, reward, new_state, alpha)
                previous_state = new_state
                i += 1


