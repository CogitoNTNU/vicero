import gym
from vicero.algorithms.deepqlearning import DQNAgent, NetworkSpecification

# This function is used to design a custom reward function, overriding 
# the one from the environment. This one rewards based on the absolute cart speed.
def state_to_reward(state):
    return abs(state[1]) * 10 - 0.1

env = gym.make('MountainCar-v0')
spec = NetworkSpecification()
agent = DQNAgent(env, spec, state_to_reward=state_to_reward, eps_decay=0.9995)

batch_size = 32
num_episodes = 1000
training_iter = 500

agent.train(num_episodes, batch_size, training_iter, verbose=True, plot=True, eps_decay=True)