import numpy as np
import pygame as pg

# The policy class is used as a simple lambda wrapper 
# to keep things a bit more clean. More functionality
# might be added in the future.

class Policy:
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(state)

class RandomPolicy(Policy):
    def __init__(self, action_space):
        super(RandomPolicy, self).__init__(lambda _ : np.random.choice(action_space))

class KeyboardPolicy(Policy):
    def __init__(self, default, up=None, down=None, left=None, right=None):
        def f(keys):
            action = default
            if keys[pg.K_UP] and up is not None:    action = up
            if keys[pg.K_DOWN] and down is not None:  action = down
            if keys[pg.K_LEFT] and left is not None: action = left
            if keys[pg.K_RIGHT] and right is not None: action = right
            return action
        
        super(KeyboardPolicy, self).__init__(f)

