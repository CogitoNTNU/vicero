from Qlearner import Qlearner
import numpy as np
import math
from collections import deque

import tensorflow as tf


def run(self):
    scores = deque(maxlen=100)

    for e in range(self.n_episodes):
        current_state = self.discretize(self.env.reset())

        alpha = self.get_alpha(e)
        epsilon = self.get_epsilon(e)
        done = False
        i = 0

        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= self.n_win_ticks and e >= 100:
            if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
            return e - 100
        if e % 100 == 0 and not self.quiet:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

    if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
    return e

def discretize(self, obs):
    upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
    lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)


if __name__ == '__main__':
    num_actions = 4
    num_states = 21
    gamma = 1
    alpha_init = 0.1
    epsilon_init = 0.1

    env = BoxEnv((5, 5), 1, 3)  # (5, 5) er size, 1 er # goals, 3 er #blocks

    method = Qlearner(num_actions, num_states, gamma, alpha_init, epsilon_init)

    while not done:
        # self.env.render()
        action = self.choose_action(current_state, epsilon)
        obs, reward, done, _ = self.env.step(action)
        new_state = self.discretize(obs)
        self.update_q(current_state, action, reward, new_state, alpha)
        current_state = new_state
        i += 1
        if e > 160:
            self.env.render()
