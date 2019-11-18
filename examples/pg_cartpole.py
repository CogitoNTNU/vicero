from vicero.algorithms.reinforce import Reinforce
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch

def plot(history):
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(history)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
    
env = gym.make('CartPole-v0')
pg = Reinforce(env, plotter=plot)
pg.train(1000)