import pygame as pg

class Overlay:
    def __init__(self, rect):
        self.rect = rect

class ActionDistributionOverlay(Overlay):
    def __init__(self, algorithm, rect):
        super(ActionDistributionOverlay, self).__init__(rect)
        self.algorithm = algorithm

    def render(self, screen, state):
        sm = self.algorithm.action_distribution(state)
        bar_w = self.rect.w // len(sm)
        
        for i in range(len(sm)):
            pg.draw.rect(screen, (0, 200, 0), pg.Rect(self.rect.x + i * bar_w, self.rect.y + self.rect.h, bar_w, -self.rect.h * sm[i]))
            pg.draw.rect(screen, (0, 160, 0), pg.Rect(self.rect.x + i * bar_w, self.rect.y + self.rect.h, bar_w, -self.rect.h * sm[i]), 1)
        

