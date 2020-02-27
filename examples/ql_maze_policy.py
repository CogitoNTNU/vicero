import pygame as pg

from examples.environments.maze import MazeEnvironment
from vicero.agent import Agent
from vicero.algorithms.qlearning import Qlearning
from vicero.policy import RandomPolicy

# In this example the focus is to see frozen policies 
# compared to each other in real time.

board = [[0, 0,  0,  0, 10,  0, 0,  0],
         [0, 0, -1, -1, -1,  0, 0,  0],
         [0, 0, -1,  0,  0, -1, 0,  0],
         [0, 0,  0,  0,  0,  0, 0, -1],
         [0, 0,  0,  0,  0,  0, 0,  0],
         [0, 0,  0,  0,  0,  0, 0,  0],
         [0, 0,  0, -1, -1,  0, 0,  0],
         [0, 0,  0,  1,  0,  0, 0,  0]]

cell_size = 48  # the size of one game cell, in pixels
framerate = 10  # frames per second

# pygame setup
pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
clock = pg.time.Clock()

env = MazeEnvironment(board, cell_size)
ql = Qlearning(env, n_states=len(board) ** 2, n_actions=len(MazeEnvironment.action_space), epsilon=0.15,
               discretize=lambda state: state[1] * env.size + state[0], table_type='numpy')


#ql = Qlearning(env, len(board) ** 2, len(MazeEnvironment.action_space), epsilon=0.15, discretize=lambda state : state[1] * env.size + state[0])

# Random for benchmarking
random_policy = RandomPolicy(MazeEnvironment.action_space)

# Train very little
ql.train(100)
policy_one = ql.copy_target_policy()

# Train some more
ql.train(3000)
policy_two = ql.copy_target_policy()

# Train way more
ql.train(20000)
policy_three = ql.copy_target_policy()

agent = Agent(env, random_policy)

policies = [('the random policy', random_policy),
            ('the bad policy', policy_one),
            ('the not good enough policy', policy_two),
            ('the good policy', policy_three)]

while True:
    for policy in policies:
        print('switching to {}'.format(policy[0]))

        agent.policy = policy[1]

        for _ in range(30):
            agent.step()
            env.draw(screen)
            pg.display.flip()
            clock.tick(framerate)
