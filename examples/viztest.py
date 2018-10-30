import pygame as pg
import examples.mazeenv as maze
import feature.Qlearner as ql

board = [[0,0,0,0,10,0,0,0],
         [0,0,-1,-1,-1,0,0,0],
         [0,0,-1,0,0,-1,0,0],
         [0,0,0,0,0,0,0,-1],
         [0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0],
         [0,0,0,-1,-1,0,0,0],
         [0,0,0,1,0,0,0,0]]

the_maze = maze.MazeEnv(board)

cell_size = 64

pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
done = False

clock = pg.time.Clock()

model = ql.Qlearner({}, num_actions=4, num_states=64)

info = {'x' : 3, 'y' : 7}

def discretize(x, y):
    return y * 8 + x

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
    dstate = discretize(info['x'], info['y'])
    ex_action = model.exploratory_action(dstate)

    board, reward, fin, info = the_maze.step(ex_action)
    
    model.update_q(dstate, ex_action, reward, discretize(info['x'], info['y']))
    screen.fill((40, 140, 40))
    
        
    for i in range(len(board[0])):
        for j in range(len(board)):
            if board[j][i] == -1:
                pg.draw.rect(screen, (64, 64, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
            if board[j][i] == 1:
                pg.draw.ellipse(screen, (180, 64, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
            if board[j][i] == 10:
                pg.draw.rect(screen, (180, 180, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
            pg.draw.rect(screen, (20, 70, 20), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size), 1)

    pg.display.flip()
    clock.tick(60)

    if fin:
        the_maze.reset()
        info = {'x' : 3, 'y' : 7}