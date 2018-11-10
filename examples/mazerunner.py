import pygame as pg
import environments.maze as maze
import vicero.models.qlearning as ql
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
n_game_ins = 5  # number of game instances
pad_cells  = 1  # padding between the visualizations
framerate  = 60 # frames per second

# pygame setup
pg.init()
screen = pg.display.set_mode((cell_size * (len(board[0]) + pad_cells) * n_game_ins, cell_size * len(board)))
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

    def discretize(self, pos):
        return pos[1] * self.env.size + pos[0]

    def game_step(self):
        # discretize current game state
        self.info['steps_taken'] += 1
        dstate = self.discretize(self.info['pos'])
        
        # let the model choose an action
        ex_action = self.model.exploratory_action(dstate)

        # run one step in the simulation
        self.board, reward, fin, self.info['pos'] = self.env.step(ex_action)
        
        # update the Q table based on the observation
        self.model.update_q(dstate, ex_action, reward, self.discretize(self.info['pos']))
        
        # visualize the new state
        self.draw_world()
        
        # if in goal state, restart
        if fin:
            self.env.reset()
            self.step_history.append(self.info['steps_taken'])
            self.info['steps_taken'] = 0
            info = {'x' : 3, 'y' : 7}

            return self.step_history,  True
        return None,  False

    def draw_world(self):
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                #qval = model.Q[discretize(i, j)][np.argmax(model.Q[discretize(i, j)])]
                qval = np.average(self.model.Q[self.discretize((i, j))])
                qval = 64 + 50 * qval
                
                pg.draw.rect(screen, (np.clip(qval, 0, 220) , 70, 20), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                
                if self.board[j][i] == -1:
                    pg.draw.rect(screen, (64, 64, 64), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 1:
                    pg.draw.ellipse(screen, (100, 24, 24), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 10:
                    pg.draw.rect(screen, (180, 180, 64), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
# For running multiple games uncomment the next three lines, but don't try to plot training statistics
#games = [GameInstance(maze.MazeEnvironment(board),
#                      ql.Qlearning(len(board) ** 2, len(maze.MazeEnvironment.action_space), epsilon=(0.05 * i)),
#                      (i * cell_size * (len(board[0]) + pad_cells), 0)) for i in range(n_game_ins)]

game = GameInstance(maze.MazeEnvironment(board), ql.Qlearning(len(board) ** 2, len(maze.MazeEnvironment.action_space), epsilon=(0.15)),
                      (2 * cell_size * (len(board[0]) + pad_cells), 0))

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