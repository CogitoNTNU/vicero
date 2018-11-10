import numpy as np

class MNKGame:
    def __init__(self, M, N, K):
        self.M = M # rows
        self.N = N # columns
        self.K = K
        self.state = np.zeros((M, N)) # [row][column] / [y][x]
        self.action_space = range(M * N)
        self.player_piece = 1

    def is_legal_action(self, state, action):
        return state[action // self.N][action % self.N] == 0

    def step(self, action):
        done = self.is_winning_move(self.state, action)
        self.state[action // self.N][action % self.N] = self.player_piece
        self.player_piece *= -1
        return done

    # run this before flipping the player piece
    def is_winning_move(self, state, action):
        y, x = action // self.N, action % self.N
        # horizontal
        length = 1

        offset = 1
        while x - offset >= 0 and self.state[y][x - offset] == self.player_piece:
            offset += 1
            length += 1

        offset = 1
        while x + offset < self.N and self.state[y][x + offset] == self.player_piece:
            offset += 1
            length += 1
        
        if length >= self.K:
            return True

        # vertical
        length = 1

        offset = 1
        while y - offset >= 0 and self.state[y - offset][x] == self.player_piece:
            offset += 1
            length += 1

        offset = 1
        while y + offset < self.M and self.state[y + offset][x] == self.player_piece:
            offset += 1
            length += 1
        
        if length >= self.K:
            return True

        # diagonal descending
        length = 1

        offset = 1
        while x - offset >= 0 and y - offset >= 0 and self.state[y - offset][x - offset] == self.player_piece:
            offset += 1
            length += 1

        offset = 1
        while x + offset < self.N and y + offset < self.M and self.state[y + offset][x + offset] == self.player_piece:
            offset += 1
            length += 1
        
        if length >= self.K:
            return True

        # diagonal ascending
        length = 1

        offset = 1
        while x - offset >= 0 and y + offset < self.M and self.state[y + offset][x - offset] == self.player_piece:
            offset += 1
            length += 1

        offset = 1
        while x + offset < self.N and y - offset >= 0 and self.state[y - offset][x + offset] == self.player_piece:
            offset += 1
            length += 1
        
        if length >= self.K:
            return True
            
        return False

ttt = MNKGame(4, 3, 3)

print(ttt.state)
print(ttt.step(2))
print(ttt.step(7))
print(ttt.step(4))
print(ttt.step(5))
print(ttt.step(6))
print(ttt.step(9))
print(ttt.state)

#for action in ttt.action_space:
#    ttt.step(action)