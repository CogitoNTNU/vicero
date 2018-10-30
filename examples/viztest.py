import pygame as pg


board = [[0,  0,  0, 10],
         [0, -1, -1,  0],
         [1,  0,  0,  0]]

cell_size = 64

pg.init()
screen = pg.display.set_mode((cell_size * len(board[0]), cell_size * len(board)))
done = False

clock = pg.time.Clock()

while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True
        
    """
    pressed = pg.key.get_pressed()
    if pressed[pg.K_UP]: y -= 3
    if pressed[pg.K_DOWN]: y += 3
    if pressed[pg.K_LEFT]: x -= 3
    if pressed[pg.K_RIGHT]: x += 3
    """ 
    
    screen.fill((40, 140, 40))
    
    for i in range(len(board[0])):
        for j in range(len(board)):
            if board[j][i] == -1:
                pg.draw.rect(screen, (64, 64, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
            if board[j][i] == 1:
                pg.draw.ellipse(screen, (180, 64, 64), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size))
            pg.draw.rect(screen, (20, 70, 20), pg.Rect(cell_size * i, cell_size * j, cell_size, cell_size), 1)

    pg.display.flip()
    clock.tick(60)