from Qlearner import Qlearner
import numpy as np
import math
from basicmaze import BasicMaze
import tensorflow as tf

def discretize(self, obs):
    upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
    lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)


if __name__ == '__main__':
    num_actions = 4
    width = 5
    height = 5
    blocks = 3
    goals = 1

    num_states = width * height - blocks - goals

    gamma = 1
    alpha_init = 0.1
    epsilon_init = 0.1
    params = {'epsilon': epsilon_init, 'alpha': alpha_init, 'gamma': gamma}

    env = BasicMaze((height, width), blocks,  goals)

    learner = Qlearner(params, num_actions, num_states, env)
    training_iter = 5000
    learner.explore(training_iter, discretize)
    

