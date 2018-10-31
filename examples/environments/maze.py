import numpy as np

class MazeEnvironment:

    # enumerated action
    UP, RIGHT, DOWN, LEFT = range(4)
    action_space = [UP, LEFT, DOWN, RIGHT]
    
    def __init__(self, board, wall_penalty = -2, time_penalty = -1):
        # setup 
        self.init_board = np.array(board)
        self.board = np.array(board)
        self.size = len(board)
        for i in range(self.size):
            for j in range(self.size):
                if (board[i][j] == 1):
                    self.init_pos = (j, i)
                    self.pos = self.init_pos

        # reward specification
        self.wall_penalty = wall_penalty
        self.time_penalty = time_penalty

    def step(self, action):
        x, y = self.pos
        
        # check for out-of-bound collisions
        if action == self.UP and y == 0:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        elif action == self.RIGHT and x == self.size - 1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        elif action == self.DOWN and y == self.size - 1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        elif action == self.LEFT and x == 0:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }

        # check for wall block collisions
        if action == self.UP and self.board[y - 1][x] == -1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        if action == self.RIGHT and self.board[y][x + 1] == -1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        if action == self.DOWN and self.board[y + 1][x] == -1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }
        if action == self.LEFT and self.board[y][x - 1] == -1:
            return self.board, self.wall_penalty, False, { 'pos': self.pos }

        # clear the current position
        self.board[y][x] = 0

        # move the player
        if action == self.UP:    y -= 1
        if action == self.RIGHT: x += 1
        if action == self.DOWN:  y += 1
        if action == self.LEFT:  x -= 1
        
        # check for goal state
        if self.board[y][x] == 10:
            return self.board, 10, True, { 'pos': self.pos }

        # place the player marker
        self.board[y][x] = 1
        self.pos = (x, y)

        return self.board, self.time_penalty, False, { 'pos': self.pos }

    def reset(self):
        self.board = self.init_board
        self.pos = self.init_pos

    def get_board(self):
        return self.board

    def get_pos(self):
        return self.pos