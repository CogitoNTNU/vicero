import numpy as np
import math
import pydot

class MCTS:
    # This is an implementation of MCTS that is focused on readability over performance

    class Node:
        def __init__(self, parent, action, env_output):
            self.action = action
            self.parent = parent
            self.state, self.done = env_output
            self.children = []
            self.visits = 0
            self.wins = 0

        def policy_value(self, opponent):
            qsa = self.wins / (1 + self.visits)
            usa = math.sqrt(math.log(self.parent.visits) / (1 + self.visits))
            return qsa + (usa if not opponent else -usa)

        quality = property(fget=lambda self : self.wins / (1 + self.visits))
    
    def __init__(self, env, M):
        self.env = env
        self.root = self.Node(None, None, (env.state, False))
        self.M = M
        self.player_id = 0#player_id # to allow self-play, instead of hard-coding player 1

    def pick_action(self, state, viz=False): # M full simulation runs
        self.root = self.Node(None, None, (state, False))
        self.player_id = state[0]

        for _ in range(self.M):
            self.tree_search(self.root)
        
        if viz:
            self.visualize_tree()

        picked_node = self.root.children[np.argmax([child.quality for child in self.root.children])]
        
        return picked_node.action

    # return the index of the child to be chosen, according to the tree policy
    def choose_child(self, node):
        if node.state[0] == self.player_id:
            return np.argmax([child.policy_value(False) for child in node.children])
        return np.argmin([child.policy_value(True) for child in node.children])

    def tree_search(self, node):
        if len(node.children):
            self.tree_search(node.children[self.choose_child(node)])
        else:
            self.node_expansion(node)

    def node_expansion(self, node):
        if not node.done:
            node.children = [self.Node(node, action, self.env.simulate(node.state, action)) for action in self.env.action_space if self.env.is_legal_action(node.state, action)]
        self.leaf_evaluation(node)

        #for child in node.children:
        #    self.leaf_evaluation(child)

    def default_policy(self, state=None):
        return np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])

    def leaf_evaluation(self, node):
        state, done = node.state, node.done
        while not done:
            state, done = self.env.simulate(state, self.default_policy(state))
        win = (self.env.get_winner(state) == self.player_id)
        self.backpropagation(node, win)

    def backpropagation(self, node, win):
        node.visits += 1
        if win:
            node.wins += 1
        if node.parent:
            self.backpropagation(node.parent, win)

    def build_graph(self, graph_root, tree_root, graph):
        node = pydot.Node(id(tree_root), style='filled', fillcolor='#{:02x}6930'.format(int(tree_root.quality * 255), int(tree_root.quality * 255)), label=str(tree_root.state[1])+',Q={:.3f}'.format(tree_root.quality))
        graph.add_node(node)
        for child in tree_root.children:
            self.build_graph(node, child, graph)
        if graph_root:
            graph.add_edge(pydot.Edge(graph_root, node, label=str(tree_root.action)))

    def visualize_tree(self):
        graph = pydot.Dot(graph_type='graph')
        self.build_graph(None, self.root, graph)
        graph.write_png('graph.png')