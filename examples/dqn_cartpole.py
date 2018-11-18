
import gym
from vicero.algorithms.deepqlearning import DQNAgent, NetworkSpecification

class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def play(self, n=-1):
        state = self.env.reset()
        while n != 0:
            action = self.policy(state)
            state, _, done, _ = self.env.step(action)
            self.env.render()
            if done: self.env.reset()
            n -= 1

env = gym.make('CartPole-v1')

spec = NetworkSpecification()
dqn = DQNAgent(env, spec, render=False)

batch_size = 32
num_episodes = 4
training_iter = 500
completion_reward = -10

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing poorly trained agent.')
policy = dqn.copy_target_policy()
agent_a = Agent(env, policy)
agent_a.play(500)

print('training...')
dqn.train(num_episodes * 5, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing better agent.')
policy = dqn.copy_target_policy()
agent_b = Agent(env, policy)
agent_b.play(500)

print('playing poorly trained agent again.')
agent_a.play(500)

env.close()