
import gym
from vicero.algorithms.deepqlearning import DQNAgent


env = gym.make('CartPole-v1')

agent = DQNAgent(env)

batch_size = 32

num_episodes = 1000

training_iter = 500

completion_reward = -10

agent.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)



