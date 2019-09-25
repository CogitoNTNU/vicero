import numpy as np
import matplotlib.pyplot as plt
from environments.gridworld import GridWorld
from vicero.algorithms.qlearning import Qlearning

world_shape = (12, 16)
goal_positions = [(2, 3), (10, 10)]

gw = GridWorld(world_shape, goal_positions)
ql = Qlearning(gw, len(gw.state_space), len(gw.action_space), gamma=0.6)

ql.train(50000)

maxq = [max(cell) for cell in ql.Q]
maxq = np.reshape(maxq, world_shape)
plt.matshow(maxq)
plt.show()