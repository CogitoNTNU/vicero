import pygame as pg
import environments.maze as maze
from vicero.algorithms.qlearning import Qlearning
import numpy as np

np.random.seed()

board = [[0 ,0 ,0 ,0 ,10,0 ,0 ,0 ],
         [0 ,0 ,-1,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,-1,0 ,0 ,-1,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,-1],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]]

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

env = maze.MazeEnvironment(board)

ql = Qlearning(env, len(board) ** 2, len(maze.MazeEnvironment.action_space), epsilon=0.15, discretize=lambda state : state[1] * env.size + state[0])

ql.train(100)
better_policy = ql.copy_target_policy()

ql.train(3000)
decent_policy = ql.copy_target_policy()

ql.train(10000)
pro_policy = ql.copy_target_policy()

game = GameInstance(env, better_policy)

policies = [('the bad policy', better_policy),
            ('the not good enough policy', decent_policy),
            ('the good policy', pro_policy)]

while True:
    for policy in policies:
        print('switching to {}'.format(policy[0]))
        
        game.policy = policy[1]
        for _ in range(30):
            game.game_step()
            pg.display.flip()
            clock.tick(framerate)