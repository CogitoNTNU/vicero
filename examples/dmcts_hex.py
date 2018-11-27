from environments.hex import HexSim
from vicero.algorithms.deepmcts import DeepMCTS
from vicero.algorithms.common.neuralnetwork import NetworkSpecification
import matplotlib.pyplot as plt

env = HexSim(5, 1)
spec=NetworkSpecification(hidden_layer_sizes=[64, 32, 16])

M = 50
mcts = DeepMCTS(env, M, epsilon=0.8, spec=spec)
mcts.train(10)
print('saving anet after {} training iterations'.format(mcts.training_iterations))
mcts.save('hex_agent.pkl')

plt.plot(mcts.loss_history)
plt.show()