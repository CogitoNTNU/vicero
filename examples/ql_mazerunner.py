import pygame as pg
from environments.maze import MazeEnvironment
from vicero.algorithms.qlearning import Qlearning
import numpy as np
import matplotlib.pyplot as plt

np.random.seed()

board = [[0 ,0 ,0 ,0 ,10,0 ,0 ,0 ],
         [0 ,0 ,-1,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,-1,0 ,0 ,-1,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,-1],
         [10,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
         [0 ,0 ,0 ,-1,-1,0 ,0 ,0 ],
         [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ]]

cell_size  = 32 # the size of one game cell, in pixels
pad_cells  = 1  # padding between the visualizations
framerate  = 60 # frames per second

# pygame setup
pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
clock = pg.time.Clock()

class GameInstance:
    def __init__(self, env, model, offset):
        self.env = env
        self.model = model
        self.offset = offset
        # steps_taken is the number of steps taken before a completion of the maze
        # Iteration is the number of times the agent has completed the environment
        self.info = {'pos': (3, 7), 'steps_taken': 0}
        self.step_history = []

    def get_epsilon(self):
        return self.model.get_epsilon()

    def game_step(self):
        # discretize current game state
        self.info['steps_taken'] += 1
        
        state = self.env.state
        
        # let the model choose an action
        action = self.model.exploratory_action(state)

        # run one step in the simulation
        new_state, reward, done, self.board = self.env.step(action)
        
        # update the Q table based on the observation
        self.model.update_q(state, action, reward, new_state)
        
        # visualize the new state
        self.draw_world()
        
        # if in goal state, restart
        if done:
            self.env.reset()
            self.step_history.append(self.info['steps_taken'])
            self.info['steps_taken'] = 0
            info = {'x' : 3, 'y' : 7}

            return self.step_history,  True
        return None,  False

    def draw_world(self):
        heatmap = np.ndarray(self.board.shape)
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                qval = np.average(self.model.Q[discretize((i, j))])
                heatmap[i][j] = 64 + 50 * qval
                
        self.env.draw(screen, heatmap)

env = MazeEnvironment(board, cell_size)

def discretize(state):
    return state[1] * env.size + state[0]

ql = Qlearning(env, len(board) ** 2, len(MazeEnvironment.action_space), epsilon=0.1, discretize=discretize)

game = GameInstance(env, ql, (0, 0))

def plot_durations(steps):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Iteration')
    plt.ylabel('Steps')
    plt.plot(steps)

    plt.pause(0.001) # To update plots

while True:
    i = 0
    step_list, finished = game.game_step()
    if finished:
        plot_durations(step_list)
    pg.display.flip()
    clock.tick(framerate)