import pygame as pg
from environments.maze import MazeEnvironment
from vicero.algorithms.qlearning import Qlearning
from vicero.policy import RandomPolicy
import numpy as np

# In this example the focus is to see frozen policies 
# compared to each other in real time.

np.random.seed()

board = [[0 ,0 ,0 ,0 ,10,0 ,0 ,0 ],
         [0 ,0 ,-1,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,-1,0 ,0 ,-1,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,-1],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]]

# TODO: move visualization etc. to the MazeEnvironment class..

cell_size  = 48 # the size of one game cell, in pixels
framerate  = 10 # frames per second

# pygame setup
pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
clock = pg.time.Clock()

class GameInstance:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def game_step(self):
        action = self.policy(self.env.state)
        _, _, done, self.board = self.env.step(action)
        self.draw_world()
        if done: self.env.reset()

    def draw_world(self):
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                
                pg.draw.rect(screen, (20, 70, 20), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
                
                if self.board[j][i] == -1:
                    pg.draw.rect(screen, (64, 64, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 1:
                    pg.draw.ellipse(screen, (100, 24, 24), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 10:
                    pg.draw.rect(screen, (180, 180, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))

env = MazeEnvironment(board)

ql = Qlearning(env, len(board) ** 2, len(MazeEnvironment.action_space), epsilon=0.15, discretize=lambda state : state[1] * env.size + state[0])

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

game = GameInstance(env, random_policy)

policies = [('the random policy',          random_policy),
            ('the bad policy',             policy_one),
            ('the not good enough policy', policy_two),
            ('the good policy',            policy_three)]

while True:
    for policy in policies:
        print('switching to {}'.format(policy[0]))
        
        game.policy = policy[1]

        for _ in range(30):
            game.game_step()
            pg.display.flip()
            clock.tick(framerate)