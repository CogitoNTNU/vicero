import vicero.algorithms.mcts as mcts
import environments.mnkgame as mnkgame
import numpy as np
import pygame as pg


M, N, K = 3, 3, 3
cell_size, framerate = 32, 2
pg.init()
screen = pg.display.set_mode((cell_size * M, cell_size * N))
clock = pg.time.Clock()

class GameAgent:
    def __init__(self, env, name, algorithm=None):
        self.env = env
        self.algorithm = algorithm
        self.name = name
    
    def pick_action(self, state, viz=False):
        if self.algorithm:
            return self.algorithm.pick_action(state, viz)
        return np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])

color_dict = {
    -1 : (255, 0, 0),
    0  : (64, 64, 64),
    1  : (0, 0, 255)
}

env = mnkgame.MNKGame(M, N, K) # tic tac toe

def match(agent1, agent2, n):
    agent1_wins = 0
    agent2_wins = 0
    
    for i in range(n_games):
        env.reset(starting_piece=1) 

        state, done = env.state, False
        
        while not done:
            if state[0] == 1:
                action = agent1.pick_action(env.state)
            else:
                action = agent2.pick_action(env.state)
            
            state, done = env.step(action)
            _, board, _ = state

            for i in range(len(board[0])):
                for j in range(len(board)):
                    pg.draw.rect(screen, color_dict[board[i][j]], pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
                                    
            pg.display.flip()
            clock.tick(framerate)
            
        if env.get_winner(state) == 1:
            agent1_wins += 1
        if env.get_winner(state) == -1:
            agent2_wins += 1

    return agent1_wins, n - agent1_wins - agent2_wins, agent2_wins

carlo =     GameAgent(env, 'Carlo Supreme', algorithm=mcts.MCTS(env, 100))
montezuma = GameAgent(env, 'Montezuma', algorithm=mcts.MCTS(env, 25))
gambler =   GameAgent(env, 'Gambler Gabe')

n_games = 100

matchups = [(carlo, carlo), 
            (carlo, montezuma),
            (carlo, gambler),
            (montezuma, gambler),
            (gambler, carlo), 
            (gambler, gambler)]

for matchup in matchups:
    w1, ties, w2 = match(*matchup, n_games)
    print('{} vs. {}, {}/{} ties={}'.format(matchup[0].name, matchup[1].name, w1, w2, ties))