import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from environments.maze import MazeEnvironment
from vicero.algorithms.qlearning import Qlearning
from vicero.agent import Agent

board = [[0  ,0  ,0  ,0  ,10 ,0  ,0  ,0  ],
         [0  ,0  ,-1 ,-1 ,-1 ,0  ,0  ,0  ],
         [0  ,0  ,-1 ,0  ,0  ,-1 ,0  ,0  ],
         [0  ,0  ,0  ,0  ,0  ,0  ,0  ,-1 ],
         [10 ,0  ,0  ,0  ,0  ,0  ,0  ,0  ],
         [0  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ],
         [0  ,0  ,0  ,-1 ,-1 ,0  ,0  ,0  ],
         [0  ,0  ,0  ,1  ,0  ,0  ,0  ,0  ]]

board = np.array(board)

cell_size  = 48 # the size of one game cell, in pixels
pad_cells  = 1  # padding between the visualizations
framerate  = 15 # frames per second

# pygame setup
pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
clock = pg.time.Clock()

class GameInstance:
    def __init__(self, env, algorithm):
        self.env = env
        self.algorithm = algorithm
        self.steps_taken = 0
        self.step_history = []

    def game_step(self):
        self.steps_taken += 1
        
        state = self.env.state
        action = self.algorithm.exploratory_action(state)
        new_state, reward, done, self.board = self.env.step(action)
        self.algorithm.update_q(state, action, reward, new_state)
        
        if done:
            self.env.reset()
            self.step_history.append(self.steps_taken)
            self.steps_taken = 0
        
            return self.step_history,  True
        return None,  False

env = MazeEnvironment(board, cell_size)

def discretize(state):
    return state[1] * env.size + state[0]

ql = Qlearning(env, len(board) ** 2, len(MazeEnvironment.action_space), epsilon=0.1, discretize=discretize)
game = GameInstance(env, ql)

def plot_durations(steps):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.plot(steps)

    plt.pause(0.001) # To update plots

while True:
    i = 0
    step_list, done = game.game_step()
    if done:
        plot_durations(step_list)
    
    heatmap = np.ndarray(board.shape)
    for i in range(len(board[0])):
        for j in range(len(board)):
            qval = np.average(ql.Q[discretize((i, j))])
            heatmap[i][j] = 64 + 50 * qval

    env.draw(screen, heatmap)

    pg.display.flip()
    clock.tick(framerate)