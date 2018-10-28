import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
# taken from https://stackoverflow.com/questions/12836191/
# randomly-generating-3-tuples-with-distinct-elements-in-python


class BasicMaze():
    def __init__(self, shape, num_boxes, num_goals):
        # black tiles: there is a box there ('k')
        # green tiles: there is a goal there ('g')
        # white tiles: possible to be there ('w')

        self.maze = np.zeros(shape, 'U1')
        self.maze.fill('w')
        self.maze[random.sample((range(0, shape[0] + 1), range(0, shape[1] + 1)), num_boxes)] = 'k'
        self.maze[random.choice(np.argwhere(self.maze != 'k'), num_goals)] = 'g'

    def visualize(self):
        