import numpy as np
import pygame as pg
from pathlib import Path
import random
mod_path = Path(__file__).parent

cheese_color = (255, 212, 0)

class MazeEnvironment:

    # enumerated action
    UP, RIGHT, DOWN, LEFT = range(4)
    action_space = [UP, LEFT, DOWN, RIGHT]
    
    def __init__(self, board, cell_size, wall_penalty=-2, time_penalty=-1):
        # setup 
        self.init_board = np.array(board)
        self.board = np.array(board)
        self.size = len(board)
        self.cell_size = cell_size

        self.free_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if (board[i][j] == 0):
                    self.free_cells.append((j, i))
                    #self.init_pos = (j, i)
                    self.state = random.choice(self.free_cells)#self.init_pos
        
        self.img_rat = pg.image.load(str(mod_path) + '/rat.png')
        self.img_rat = pg.transform.scale(self.img_rat, (cell_size, cell_size))

        # reward specification
        self.wall_penalty = wall_penalty
        self.time_penalty = time_penalty

    def step(self, action):
        x, y = self.state
        
        # check for out-of-bound collisions
        if action == self.UP and y == 0:
            return self.state, self.wall_penalty, False, self.board
        elif action == self.RIGHT and x == self.size - 1:
            return self.state, self.wall_penalty, False, self.board
        elif action == self.DOWN and y == self.size - 1:
            return self.state, self.wall_penalty, False, self.board
        elif action == self.LEFT and x == 0:
            return self.state, self.wall_penalty, False, self.board

        # check for wall block collisions
        if action == self.UP and self.board[y - 1][x] == -1:
            return self.state, self.wall_penalty, False, self.board
        if action == self.RIGHT and self.board[y][x + 1] == -1:
            return self.state, self.wall_penalty, False, self.board
        if action == self.DOWN and self.board[y + 1][x] == -1:
            return self.state, self.wall_penalty, False, self.board
        if action == self.LEFT and self.board[y][x - 1] == -1:
            return self.state, self.wall_penalty, False, self.board

        # clear the current position
        self.board[y][x] = 0

        # move the player
        if action == self.UP:    y -= 1
        if action == self.RIGHT: x += 1
        if action == self.DOWN:  y += 1
        if action == self.LEFT:  x -= 1
        
        # check for goal state
        if self.board[y][x] == 10:
            return self.state, 1, True, self.board

        # place the player marker
        self.board[y][x] = 1
        self.state = (x, y)

        return self.state, self.time_penalty, False, self.board

    def reset(self):
        self.board = np.array(self.init_board)
        self.state = random.choice(self.free_cells)
        x, y = self.state
        self.board[y][x] = 1

    def randomize(self):
        self.board = np.array(self.init_board)
        x, y = (0, 0)
        free = False
        while not free:
            x, y = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if self.board[y][x] == 0:
                free = True
        self.board[y][x] = 1

    def draw(self, screen, heatmap=None):
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                cell = pg.Rect(self.cell_size * i, self.cell_size * j, self.cell_size, self.cell_size)
                
                if heatmap is not None:
                    pg.draw.rect(screen, (np.clip(heatmap[i][j], 0, 220) , 70, 20), cell)
                else:
                    pg.draw.rect(screen, (64, 64, 64), cell)

                if self.board[j][i] == -1: 
                    pg.draw.rect(screen, (48, 48, 48), cell)
                    pg.draw.rect(screen, (16, 16, 16), cell, 1)
                if self.board[j][i] == 1:  screen.blit(self.img_rat, cell)
                if self.board[j][i] == 10: pg.draw.rect(screen, cheese_color, cell)
                
    def get_board(self):
        return self.board

    def get_pos(self):
        return self.state

from numpy.random import randint as rand
#import matplotlib.pyplot as pyplot

def maze(width=81, height=51, complexity=.75, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) # number of components
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) # size of components
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2 # pick a random position
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z

#pyplot.figure(figsize=(10, 5))
#pyplot.imshow(maze(80, 40), cmap=pyplot.cm.binary, interpolation='nearest')
#pyplot.xticks([]), pyplot.yticks([])
#pyplot.show()