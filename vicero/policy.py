# The policy class is used as a simple lambda wrapper 
# to keep things a bit more clean. More functionality
# might be added in the future.

class Policy:
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(state)