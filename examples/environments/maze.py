import numpy as np
import pygame as pg

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
        for i in range(self.size):
            for j in range(self.size):
                if (board[i][j] == 1):
                    self.init_pos = (j, i)
                    self.state = self.init_pos

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
            return self.state, 10, True, self.board

        # place the player marker
        self.board[y][x] = 1
        self.state = (x, y)

        return self.state, self.time_penalty, False, self.board

    def reset(self):
        self.board = np.array(self.init_board)
        self.state = self.init_pos
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
                    pg.draw.rect(screen, (20, 70, 20), cell)

                if self.board[j][i] == -1: pg.draw.rect(screen, (64, 64, 64), cell)
                if self.board[j][i] == 1:  pg.draw.ellipse(screen, (100, 24, 24), cell)
                if self.board[j][i] == 10: pg.draw.rect(screen, (180, 180, 64), cell)
                
    def get_board(self):
        return self.board

    def get_pos(self):
        return self.state