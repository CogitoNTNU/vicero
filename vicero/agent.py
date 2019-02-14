import vicero.policy

class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def step(self):
        action = self.policy(self.env.state)
        _, _, done, _ = self.env.step(action)
        if done: self.env.reset()    