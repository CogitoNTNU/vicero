import random

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg

from examples.environments.maze import maze, MazeEnvironment
from vicero.algorithms.qlearning import Qlearning
from vicero.visualization.overlay import ActionDistributionOverlay as ADO

gen = maze(32, 32)

board = np.zeros(gen.shape)

free_cells = []
for row in range(board.shape[0]):
    for col in range(board.shape[1]):
        if gen[row][col]:
            board[row][col] = -1
        else:
            free_cells.append((row, col))

random.shuffle(free_cells)
board[free_cells[0][0]][free_cells[0][1]] = 10

cell_size = 24  # the size of one game cell, in pixels
pad_cells = 1  # padding between the visualizations
framerate = 500  # frames per second

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
        action = self.algorithm.action(state)
        new_state, reward, done, self.board = self.env.step(action)
        self.algorithm.update_q(state, action, reward, new_state)

        if done:
            self.env.reset()
            self.step_history.append(self.steps_taken)
            self.steps_taken = 0

            return self.step_history, True
        return None, False


env = MazeEnvironment(board, cell_size)


def discretize(state):
    return state[1] * env.size + state[0]


ql = Qlearning(env, n_states=len(board) ** 2, n_actions=len(MazeEnvironment.action_space), epsilon=0.1, gamma=0.8,
               discretize=discretize, table_type='numpy')
game = GameInstance(env, ql)


def plot_durations(steps):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.plot(steps)

    plt.pause(0.001)  # To update plots


ado = ADO(ql, pg.Rect(screen.get_width() - 128, screen.get_height() - 128, 128, 128))

for _ in range(500000):
    step_list, done = game.game_step()

while True:
    i = 0
    step_list, done = game.game_step()
    # if done:
    #    plot_durations(step_list)

    heatmap = np.ndarray(board.shape)
    for i in range(len(board[0])):
        for j in range(len(board)):
            qval = np.average(ql.Q[discretize((i, j))])
            heatmap[i][j] = 64 + 50 * qval

    env.draw(screen)
    ado.render(screen, game.env.state)

    pg.display.flip()
    if framerate > 10:
        framerate = framerate * 0.999

    clock.tick(int(framerate))
