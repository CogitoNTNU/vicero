import numpy as np
import math
import pydot as pd

# player id is 1 (blue) or 2 (red)
# player 1 wins by bridging top-left to bottom-right or vice versa (on the visualized diamond form)
# player 2 wins by bridging top-right to bottom-left or vice versa  (on the visualized diamond form)

class HexSim:
    class Node:
        def __init__(self, list_index):
            self.list_index = list_index
            self.neighbors = []
            self.border_identities = []

        def __repr__(self):
            return str(self.list_index)

        def init_edges(self, game_graph):
            N = int(math.sqrt(len(game_graph)))
            nr = range(N)
            r, c = int(self.list_index // N), int(self.list_index % N)
            
            def rc_i(row, col):
                return row * N + col

            if r + 1 in nr: self.neighbors.append(game_graph[rc_i(r + 1, c)])
            if r - 1 in nr: self.neighbors.append(game_graph[rc_i(r - 1, c)])
            if c + 1 in nr: self.neighbors.append(game_graph[rc_i(r, c + 1)])
            if c - 1 in nr: self.neighbors.append(game_graph[rc_i(r, c - 1)])
            if r - 1 in nr and c + 1 in nr: self.neighbors.append(game_graph[rc_i(r - 1, c + 1)])
            if r + 1 in nr and c - 1 in nr: self.neighbors.append(game_graph[rc_i(r + 1, c - 1)])
            
            if r == 0: self.border_identities.append(-2)
            if c == 0: self.border_identities.append(-1)
            if r == N - 1: self.border_identities.append(2)
            if c == N - 1: self.border_identities.append(1)
            
    def __init__(self, N, starting_player):
        self.N = N
        self.board = np.zeros(self.N * self.N)

        # list of references to all the nodes, nodes contain info about neighbor cells
        self.game_graph = [HexSim.Node(i) for i in range(self.N * self.N)]
        
        # after all nodes exist, connect them
        for node in self.game_graph:
            node.init_edges(self.game_graph)

        self.action_space = [i for i in range(self.N * self.N)]
        self.next_player_id = starting_player
        
        # number of players, defined to avoid constants in the code
        self.P = 2
        
        self.flat_state_size = N * N + 1
        self.state_flattener = lambda state : np.array([state[0]] + list(state[1]))
        
    state = property(fget=lambda self : (self.next_player_id, self.board))

    def is_winning_move(self, state, action):
        player, board = state

        def bfs(graph, start):
            visited, queue = set(), [start]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    for neighbor in node.neighbors:
                        if board[neighbor.list_index] == player:
                            queue.append(neighbor)
                    visited.add(node)
            return visited

        connected_nodes = bfs(self.game_graph, self.game_graph[action])
        
        ends = 0
        for node in connected_nodes:
            if player in node.border_identities:
                ends += 1
                break

        for node in connected_nodes:
            if player * -1 in node.border_identities:
                ends += 1
                break

        return ends == 2
        
    # one step in the actual game, assumes legal action
    def step(self, action):
        self.board[action] = self.next_player_id
        done = self.is_winning_move(self.state, action)
        self.next_player_id = 1 + ((self.next_player_id + 2) % self.P)
        return self.state, done

    def simulate(self, state, action):
        player, board = state
        board = np.array(board)
        board[action] = player
        done = self.is_winning_move(state, action)
        player = 1 + ((player + 2) % self.P)
        return (player, board), done
    
    def is_legal_action(self, state, action):
        if action not in self.action_space:
            return False
        return state[1][action] == 0

    def reset(self, starting_player):
        self.board = np.zeros(self.N * self.N)
        self.next_player_id = starting_player
        return self.state

    def get_winner(self, state):
        return 1 + ((state[0] + 2) % self.P)
        

    def visualize(self, fname):
        color_map = {
            0: 'gray',
            1: 'blue',
            2: 'red'
        }
        
        graph = pd.Dot(graph_type='graph', nodesep=0.3)
        
        for node in self.game_graph:
            pd_node = pd.Node(node.list_index, shape='hexagon', style='filled', fillcolor=color_map[self.board[node.list_index]])#, label='{}, {}, {}'.format(hnode.listform_index, hnode.color, hnode.connection_end))
            graph.add_node(pd_node)
            for neighbor in node.neighbors:
                graph.add_edge(pd.Edge(pd_node, neighbor.list_index))

        graph.write_png(fname, prog='neato')
# test
"""
hs = HexSim(5, 1)

for action in hs.action_space:
    assert hs.is_legal_action(hs.state, action) == True

for action in hs.action_space:
    hs.board[action] = 1
    assert hs.is_legal_action(hs.state, action) == False

for action in hs.action_space:
    assert hs.is_legal_action(hs.state, action) == False

player = 1
hs.reset(player)
for action in hs.action_space:
    assert hs.is_legal_action(hs.state, action) == True
    new_state, done = hs.step(action)
    assert new_state[0] != player
    player = new_state[0]


state, done = hs.reset(player), False
while not done:
    hs.visualize('before_win.png')
    action = np.random.choice([action for action in hs.action_space if hs.is_legal_action(state, action)])
    state, done = hs.step(action)

print(hs.get_winner(state))
hs.visualize('win.png')
"""