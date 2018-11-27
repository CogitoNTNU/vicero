import numpy as np
import math
import pydot
import torch
import torch.nn as nn
from copy import deepcopy
from vicero.algorithms.common.neuralnetwork import NeuralNetwork, NetworkSpecification
from vicero.algorithms.common.replay_buffer import ReplayBuffer

class DeepMCTS:

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
    
    def __init__(self, env, M, minibatch_size=32, epsilon=0.2, alpha=0.00001, anet_path=None, loss_fct=nn.MSELoss, optimizer=torch.optim.Adam, spec=NetworkSpecification()):
        self.env = env
        self.root = self.Node(None, None, (env.state, False))
        self.M = M
        self.player_id = 0
        self.replay_buffer = ReplayBuffer()
        
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.anet = NeuralNetwork(self.env.flat_state_size, len(self.env.action_space), spec=spec)
        if anet_path is not None:
            self.anet.load_state_dict(torch.load(anet_path))
            
        self.device = torch.device('cpu')
        self.minibatch_size = minibatch_size

        self.alpha = alpha
        self.epsilon = epsilon
        self.optimizer = optimizer(self.anet.parameters(), lr=self.alpha)
        self.loss_fct = loss_fct()
        
        self.loss_history = []
        self.training_iterations = 0

    def pick_action(self, state, viz=False): # M full simulation runs
        if (self.root.state[0] != state[0]) or not (list(self.root.state[1]) == list(state[1])):
            self.root = self.Node(None, None, (state, False))
        self.player_id = state[0]

        for _ in range(self.M):
            self.tree_search(self.root)
        
        if viz:
            self.visualize_tree()

        distribution = np.zeros(len(self.env.action_space))
        max_visits = 0
        for node in self.root.children:
            if node.visits > max_visits:
                max_visits = node.visits
                
        for node in self.root.children:
            distribution[node.action] = node.visits / (max_visits + 1)
        self.replay_buffer.remember(state, distribution)
        
        if len(self.replay_buffer.buffer) >= self.minibatch_size:
            self.replay()

        state_tensor = torch.from_numpy(self.env.state_flattener(state))
        state_tensor = state_tensor.to(self.device)
        anet_action = self.anet(state_tensor).max(0)[1].numpy()
        
        chosen_action = np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])
        if self.env.is_legal_action(state, anet_action) and np.random.uniform() > self.epsilon:
            chosen_action = anet_action
        for child in self.root.children:
            if child.action == chosen_action:
                self.root = child
        return chosen_action

        #picked_node = self.root.children[np.argmax([child.quality for child in self.root.children])]
        #return picked_node.action

    # return the index of the child to be chosen, according to the tree policy
    def choose_child(self, node):
        if node.state[0] == self.player_id:
            return np.argmax([child.policy_value(False) for child in node.children])
        return np.argmin([child.policy_value(True) for child in node.children])

    def save(self, name):
        torch.save(self.anet.state_dict(), name)

    def tree_search(self, node):
        if len(node.children):
            self.tree_search(node.children[self.choose_child(node)])
        else:
            self.node_expansion(node)

    def train(self, n):
        for i in range(n):
            print('{:03}/{:03}'.format(i, n))
            state, done = self.env.reset(1 if i % 2 == 0 else 2), False
            while not done:
                if state[0] == 1:
                    action = self.pick_action(state)
                else:
                    action = self.pick_action(state)
                state, done = self.env.step(action)

    def replay(self):
        total_loss = 0
        for _ in range(self.minibatch_size):
            state, distribution = self.replay_buffer.sample()
            distribution = torch.tensor(distribution, dtype=torch.double, requires_grad=False)

            state_tensor = torch.from_numpy(self.env.state_flattener(state))
            state_tensor = state_tensor.to(self.device)
            anet_action = self.anet(state_tensor).max(0)[1].numpy()
                
            #target_f = self.anet(state_tensor)
            #target_f[anet_action] = distribution[anet_action]
            prediction = self.anet(state_tensor)

            loss = self.loss_fct(prediction, distribution)
            total_loss += float(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_iterations += 1
        self.loss_history.append(total_loss / self.minibatch_size)
    
    def node_expansion(self, node):
        if not node.done:
            node.children = [self.Node(node, action, self.env.simulate(node.state, action)) for action in self.env.action_space if self.env.is_legal_action(node.state, action)]
        self.leaf_evaluation(node)

        #for child in node.children:
        #    self.leaf_evaluation(child)

    def target_policy(self, state):
        state_tensor = torch.from_numpy(self.env.state_flattener(state))
        state_tensor = state_tensor.to(self.device)
        distribution = self.anet(state_tensor) 
        
        tries = 0
        anet_action = distribution.max(0)[1].numpy()    
        while not self.env.is_legal_action(state, anet_action):
            distribution[anet_action] = 0
            anet_action = distribution.max(0)[1].numpy()
            tries += 1
            if tries > len(self.env.action_space) // 2:
                anet_action = np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])

        return anet_action
        
        #return np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])

    def default_policy(self, state):
        state_tensor = torch.from_numpy(self.env.state_flattener(state))
        state_tensor = state_tensor.to(self.device)
        anet_action = self.anet(state_tensor).max(0)[1].numpy()
        
        if self.env.is_legal_action(state, anet_action):
            return anet_action
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
    
    def copy_target_policy(self):
        cpy = deepcopy(self.anet)
        device = self.device
        def policy(state):
            state_tensor = torch.from_numpy(self.env.state_flattener(state))
            state_tensor = state_tensor.to(self.device)
            anet_action = self.anet(state_tensor).max(0)[1].numpy()
            
            if self.env.is_legal_action(state, anet_action):
                return anet_action
            return np.random.choice([action for action in self.env.action_space if self.env.is_legal_action(state, action)])

        return policy