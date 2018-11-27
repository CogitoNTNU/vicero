import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def remember(self, state, distribution):
        self.buffer.append((state, distribution))
    
    def sample(self):
        return self.buffer[np.random.randint(len(self.buffer))]