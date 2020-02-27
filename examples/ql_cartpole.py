import math
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from vicero.algorithms.qlearning import Qlearning


def plotter(data):
    plt.figure(1)
    plt.clf()
    plt.title("Reward per episode over tid")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    n = 600
    if len(data) > n:

        x = np.linspace(len(data) - n, len(data), n)
        plt.plot(x, data[-n:])

    else:
        plt.plot(data)
    plt.pause(0.001)


class Discretize:

    def __init__(self, env, buckets=(1, 1, 6, 12)):
        self.env = env
        self.buckets = buckets

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)


if __name__ == '__main__':
    plt.style.use('ggplot')
    data = np.random.randn(50)
    env = gym.make('CartPole-v1')

    disc = Discretize(env)

    learning_decay = lambda lr, t: max(0.1, min(0.5, 1.0 - math.log10((t + 1) / 25)))
    epsilon_decay = lambda eps, t: max(0.01, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    agent = Qlearning(env, learning_rate=0.5, gamma=0.99, epsilon=0.1, learning_rate_decay=learning_decay,
                      epsilon_decay=epsilon_decay, table_type='dict', n_actions=env.action_space.n,
                      discretize=disc.discretize) #, plotter=plotter)

    agent.train(300)

    agent.epsilon = 0
    i_state = env.reset()
    state = i_state
    done = False
    while not done:
        env.render()
        state, _, done, _ = env.step(agent.action(state))
        time.sleep(0.01)
    env.close()
