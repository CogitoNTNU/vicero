import numpy as np

class MazeEnv:

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


    def __init__(self, board):
        self.init_board = np.array(board)
        self.board = np.array(board)
        self.action_space = [self.UP, self.LEFT, self.DOWN, self.RIGHT]
        self.size = len(board)
        for i in range(size):
            for j in range(size):
                if (board[i][j] == 1):
                    self.init_pos = (j, i)
                    self.pos = self.init_pos

    def step(self, action):
        reward = 0
        y, x = self.pos
        if action == self.UP and y == 0:
            return self.board, -10, False, {}
        elif action == self.RIGHT and x == self.size - 1:
            return self.board, -10, False, {}
        elif action == self.DOWN and y == self.size - 1:
            return self.board, -10, False, {}
        elif action == self.LEFT and y == self.size:
            return self.board, -10, False, {}

        self.board[x][y] = 0

        if action == self.UP: y += 1
        if action == self.RIGHT: x += 1
        if action == self.DOWN: y -= 1
        if action == self.LEFT: x -= 1

        if self.board[x][y] == 10:
            return self.board, 10, True, {}

        self.board[x][y] = 1
        self.pos = (x, y)
        return self.board, reward, False, {}

    def reset(self):
        self.board = self.init_board
        self.pos = self.init_pos

    def get_board(self):
        return self.board

    def get_pos(self):
        return self.pos


board = [[0, 0, 0, 0, 0],
        [0, 0, -1, 10, 0],
        [0, 0, -1, -1, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]]

maze = MazeEnv(board)
print(maze.board)
