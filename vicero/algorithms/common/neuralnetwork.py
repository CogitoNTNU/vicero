import torch
import torch.nn as nn
from functools import reduce

class NetworkSpecification:
    def __init__(self, hidden_layer_sizes=[24, 24], activation_function=nn.ReLU):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, spec=NetworkSpecification()):
        super(NeuralNetwork, self).__init__()

        self.layers = []
        
        prev_size = input_size
        for i in range(len(spec.hidden_layer_sizes)):
            hidden_layer_size = spec.hidden_layer_sizes[i]
            
            self.layers.append(nn.Linear(prev_size, hidden_layer_size))
            #self.add_module('hidden_layer' + str(i), self.layers[-1])
            
            self.layers.append(spec.activation_function())
            #self.add_module('activation_layer' + str(i), self.layers[-1])
            
            prev_size = hidden_layer_size
        
        self.layers.append(nn.Linear(prev_size, output_size))
        #self.add_module("output_layer", self.layers[-1])
        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn.forward(x)
        #return reduce(lambda t, f: f(t), [x] + self.layers)
        """
        tensor = self.layers[0](x)
        for layer in self.layers[1:]:
            tensor = layer(tensor)
        return tensor
        """