import gym
from vicero.algorithms.deepqlearning import DQN, NetworkSpecification
import matplotlib.pyplot as plt

# This function is used to design a custom reward function, overriding 
# the one from the environment. This one rewards based on the absolute cart speed.
def state_to_reward(state):
    return abs(state[1]) * 10 - 0.05

env = gym.make('MountainCar-v0')
spec = NetworkSpecification()

dqn = DQN(env, spec=spec, state_to_reward=state_to_reward)

batch_size = 32
num_episodes = 200
training_iter = 500

dqn.train(num_episodes, batch_size, training_iter, verbose=True, plot=True)

plt.plot(dqn.history)
plt.show()

plt.plot(dqn.maxq_history)
plt.show()