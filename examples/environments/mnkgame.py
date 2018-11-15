import numpy as np
import operator as op

class MNKGame:
    def __init__(self, M, N, K, starting_piece=1):
        self.M = M # rows
        self.N = N # columns
        self.K = K
        self.state = (starting_piece, np.zeros((M, N))) # [row][column] / [y][x]
        self.action_space = range(M * N)
        
    def is_legal_action(self, state, action):
        return state[1][action // self.N][action % self.N] == 0

    def step(self, action):
        done = self.is_winning_move(self.state, action)
        player, board = self.state
        board[action // self.N][action % self.N] = player
        self.state = (player * -1, board)
        return self.state, done
    
    def simulate(self, state, action):
        done = self.is_winning_move(state, action)
        player, board = state
        board_cpy = np.array(board)
        board_cpy[action // self.N][action % self.N] = player
        state = (player * -1, board_cpy)
        return state, done

    # run this before flipping the player piece
    def is_winning_move(self, state, action):
        player, board = state
        y, x = action // self.N, action % self.N
        x_range, y_range = range(0, self.N), range(0, self.M)
        
        idf = lambda x, y : x # identity function
        
        ops = [[op.sub, idf],    # horizontal
               [op.add, idf],    # horizontal
               [idf, op.sub],    # vertical
               [idf, op.add],    # vertical
               [op.sub, op.sub], # diagonal desc
               [op.add, op.add], # diagonal desc
               [op.sub, op.add], # diagonal asc
               [op.add, op.sub]] # diagonal asc
               
        length = 1
        for i in range(len(ops)):
            op_list = ops[i]
            
            offset = 1
            while op_list[0](x, offset) in x_range and \
                  op_list[1](y, offset) in y_range and \
                  board[op_list[1](y, offset)][op_list[0](x, offset)] == player:
                offset += 1
                length += 1
            
            if i % 2 == 1:
                if length >= self.K:
                    return True    
                length = 1
        
        return False
        
    def reset(self, starting_piece):
        self.state = (starting_piece, np.zeros((self.M, self.N)))

    def get_winner(self, state):
        return state[0]

"""
ttt = MNKGame(4, 3, 3)

ttt.state = (1, np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]]))
assert ttt.step(8)[1] == True

ttt.state = (1, np.array([[1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 0]]))
ttt.player_piece = 1
assert ttt.step(2)[1] == True

ttt.state = (1, np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0],
                          [0, 1, 0]]))
assert ttt.step(1)[1] == False

ttt.state = (-1, np.array([[-1, 0, 0],
                           [0, -1, 0],
                           [0,  0, 0],
                           [-1,-1, 0]]))
assert ttt.step(11)[1] == True

ttt.state = (1, np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, 0, 1],
                          [0, 0, 0]]))
assert ttt.step(2)[1] == True
"""