import torch
import torch.nn as nn
import gym
from vicero.algortithms.deepqlearning import DQNAgent

class Model(nn.Module):
    # Simple net with one hidden layer
    def __init__(self, input_size, first_layer, second_layer, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, first_layer)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(first_layer, second_layer)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(second_layer, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out



if __name__ == '__main__':
    torch.set_default_tensor_type('torch.DoubleTensor')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    env = gym.make('CartPole-v1')
    feature_size, action_size = env.observation_space.shape[0], env.action_space.n
    first_layer = 24
    second_layer = 24
    model = Model(feature_size, first_layer, second_layer, action_size)
    optimizer = torch.optim.Adam
    loss_fct = nn.MSELoss
    agent = DQNAgent(model, env, feature_size, action_size, device,
                     optimizer=optimizer, loss_fct=loss_fct)
    batch_size = 32
    num_episodes = 1000
    training_iter = 500

    completion_reward = -10

    agent.train(num_episodes, batch_size, training_iter, verbose=True,
                completion_reward=completion_reward, plot=True, eps_decay=True)



