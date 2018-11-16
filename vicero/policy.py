class Policy:
    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        return self.f(state)