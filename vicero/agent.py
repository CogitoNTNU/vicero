import vicero.policy

class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.state = self.env.reset()

        self.total_reward = 0
        self.current_return = 0
        self.n_episodes = 0
    
    performance = property(fget=lambda self : self.total_reward / (self.n_episodes) if self.n_episodes > 0 else self.current_return)

    def step(self, render=False, measure=False):
        action = self.policy(self.state)
        self.state, reward, done, _ = self.env.step(action)
        
        if render: self.env.render()
        if measure: self.current_return += reward
        
        if done:
            self.env.reset()
            if measure:
                self.total_reward += self.current_return
                self.current_return = 0
                self.n_episodes += 1