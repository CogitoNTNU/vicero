import vicero.models.mcts as mcts
import environments.mnkgame as mnkgame
import numpy as np

class GameAgent:
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
    
    def pick_action(self, state, viz=False):
        if self.model:
            return self.model.pick_action(state, viz)
        return np.random.choice(self.env.action_space)

M, N, K, MCTS_M = 3, 3, 3, 3
player_id, evil_id = 1, -1
starting_player = player_id
ttt = mnkgame.MNKGame(M, N, K, starting_piece=player_id) # tic tac toe

player_agent = GameAgent(ttt, model=mcts.MCTS(ttt, MCTS_M, player_id))
evil_agent = GameAgent(ttt, model=mcts.MCTS(ttt, MCTS_M, evil_id))

n_games = 1
wins = 0

for i in range(n_games): # for each game
    print('game', i)
    ttt.reset(starting_player) 
    done = False
    state = ttt.state
    it = 0
    while not done: # for each turn
        if state[0] == player_id: # shitty loop, but readable
            action = player_agent.pick_action(ttt.state, viz=(it==0))
        else: # opponent move
            action = evil_agent.pick_action(ttt.state)
        
        state, done = ttt.step(action)
        it += 1
        
    if state[0] != player_id:
        wins += 1

print('wins: {}/{}'.format(wins, n_games))