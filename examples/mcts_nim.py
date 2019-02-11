import vicero.algorithms.mcts as mcts
import environments.nimsim as nimsim
import numpy as np

class GameAgent:
    def __init__(self, env, algorithm=None):
        self.env = env
        self.algorithm = algorithm
    
    def pick_action(self, state, viz=False):
        if self.algorithm:
            return self.algorithm.pick_action(state, viz)
        return np.random.choice(self.env.action_space)

N, K, M = 4, 2, 1000
player_id, evil_id = 0, 1
starting_player = player_id
ns = nimsim.NimSim(N, K, starting_player=starting_player)

player_agent = GameAgent(ns, algorithm=mcts.MCTS(ns, M))
evil_agent = GameAgent(ns, algorithm=mcts.MCTS(ns, M))

n_games = 10
wins = 0

for i in range(n_games): # for each game
    print('game', i)
    ns.reset(starting_player) 
    done = False
    state = ns.state
    
    i = 0
    while not done: # for each turn
        
        if state[0] == player_id: # shitty loop, but readable
            action = player_agent.pick_action(ns.state, viz=(i == 0))
        else: # opponent move
            action = evil_agent.pick_action(ns.state)
        i += 1
        state, done = ns.step(action)
        
    if state[0] != player_id:
        wins += 1

print('wins: {}/{}'.format(wins, n_games))