class NimSim:
    # Nim Simulator Hard-coded for two players

    def __init__(self, N, K, starting_player):
        self.N = N
        self.K = K
        self.NP = 2 # Number of Players
        self.state = (starting_player, self.N)
        self.action_space = range(1, K + 1)
    
    def step(self, amount):
        player, pieces = self.state
        self.state = ((player + 1) % self.NP, pieces - amount)
        return self.state, (self.state[1] <= 0)

    def simulate(self, amount, state):
        player, pieces = state
        state = ((player + 1) % self.NP, pieces - amount)
        return state, (state[1] <= 0)

    def reset(self, starting_player):
        self.state = (starting_player, self.N)

    def is_legal_action(self, state, action):
        if action in self.action_space and \
           state[1] - action >= 0:
            return True

    def get_winner(self, state):
        return state[0]