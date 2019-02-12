from environments.hex import HexSim
from vicero.algorithms.deepmcts import DeepMCTS
from vicero.algorithms.common.neuralnetwork import NetworkSpecification
import matplotlib.pyplot as plt

env  = HexSim(5, 1)

# The spec is a way to describe the neural network used in the algorithm
spec = NetworkSpecification(hidden_layer_sizes=[64, 32, 16])

# A higher M will slow down training, but it will also improve the estimates that are being
# approximated. In other words, the accuracy will more clearly reflect quality of decisions.
M = 50

mcts = DeepMCTS(env, M, epsilon=0.8, spec=spec)
mcts.train(10)

# Save the results so you can load the exact same agent later
print('saving anet after {} training iterations'.format(mcts.training_iterations))
mcts.save('hex_agent.pkl')

plt.plot(mcts.loss_history)
plt.show()