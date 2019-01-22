from vicero.algorithms.qlearning import Qlearning
import numpy as np

# DynaQ is an algorithm that is very similar to
# classical tabular Q-learning, with the one difference being that it
# keeps an internal model that simulates the consequences of actions
# based entirely on experience

# More details: S&B18 Chapter 8

class DynaQ(Qlearning):
    def __init__(self, env, n_states, n_actions, epsilon, discretize, planning_steps=0):
        super(DynaQ, self).__init__(env, n_states, n_actions, epsilon=epsilon, discretize=discretize)
        self.model = {} # internal model for simulation, built from experience / exploration
        self.planning_steps = planning_steps

    def train(self, iterations):
        print(':D')
        for _ in range(iterations):    
            state_old = self.env.state
            action = self.exploratory_action(self.env.state)
            state, reward, done, board = self.env.step(action)
            self.update_q(state_old, action, reward, state)
            self.model[(state_old, action)] = (reward, state)

            for _ in range(self.planning_steps):
                sample_state_old, sample_action = np.random.sample(self.model.keys)
                sample_reward, sample_state = self.model((sample_state_old, sample_action))
                self.update_q(sample_state_old, sample_action, sample_reward, sample_state)
                

            if done:
                self.env.reset()