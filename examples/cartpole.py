import gym
import vicero.models.qlearning as ql

env = gym.make('CartPole-v0')
solver = ql.QCartPoleSolver(env=env)
solver.run()