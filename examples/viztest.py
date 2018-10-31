import pygame as pg
import examples.mazeenv as maze
import feature.Qlearner as ql
import numpy as np

np.random.seed()

board = [[0,0,0,0,10,0,0,0],
         [0,0,-1,-1,-1,0,0,0],
         [0,0,-1,0,0,-1,0,0],
         [0,0,0,0,0,0,0,-1],
         [10,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,-1,-1,0,0,0],
         [0,0,0,1,0,0,0,0]]

the_maze = maze.MazeEnv(board)

cell_size = 32

n_game_ins = 5
pad_cells = 1

pg.init()
screen = pg.display.set_mode((cell_size * (len(board[0]) + pad_cells) * n_game_ins, cell_size * len(board)))
done = False

clock = pg.time.Clock()

model = ql.Qlearner({}, num_actions=4, num_states=64)

def discretize(x, y):
    return y * 8 + x

class GameInstance:
    def __init__(self, env, model, offset):
        self.env = env
        self.model = model
        self.offset = offset
        self.info = {'x' : 3, 'y' : 7}
    
    def game_step(self):
        dstate = discretize(self.info['x'], self.info['y'])
        ex_action = self.model.exploratory_action(dstate)

        self.board, reward, fin, self.info = self.env.step(ex_action)
        
        self.model.update_q(dstate, ex_action, reward, discretize(self.info['x'], self.info['y']))
        self.draw_world()
        
        if fin:
            self.env.reset()
            info = {'x' : 3, 'y' : 7}

    def draw_world(self):
        #screen.fill((40, 140, 40))
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                #qval = model.Q[discretize(i, j)][np.argmax(model.Q[discretize(i, j)])]
                qval = np.average(self.model.Q[discretize(i, j)])
                qval = 64 + 50 * qval
                
                pg.draw.rect(screen, (np.clip(qval, 0, 220) , 70, 20), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                
                if self.board[j][i] == -1:
                    pg.draw.rect(screen, (64, 64, 64), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 1:
                    pg.draw.ellipse(screen, (100, 24, 24), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))
                if self.board[j][i] == 10:
                    pg.draw.rect(screen, (180, 180, 64), pg.Rect(self.offset[0] + cell_size * i, self.offset[1] + cell_size * j, cell_size, cell_size))

games = [GameInstance(maze.MazeEnv(board), ql.Qlearner({}, num_actions=4, num_states=64), (i * cell_size * (len(board[0]) + pad_cells), 0)) for i in range(n_game_ins)]

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
    """    
    pressed = pg.key.get_pressed()
    action = -1
    if pressed[pg.K_UP]:    action = maze.MazeEnv.UP 
    if pressed[pg.K_DOWN]:  action = maze.MazeEnv.DOWN
    if pressed[pg.K_LEFT]:  action = maze.MazeEnv.LEFT
    if pressed[pg.K_RIGHT]: action = maze.MazeEnv.RIGHT
    """
    for game in games:
        game.game_step()
    
    pg.display.flip()
    clock.tick(60)