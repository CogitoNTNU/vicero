import gym
from vicero.algorithms.deepqlearning import DQNAgent, NetworkSpecification

# This example is showing off multiple concepts, for a more pure
# DQN example, see mountaincar. The first part is obviously to
# solve cartpole, but in addition, the script will save the policy
# as it is after one shorter round of training, then it will train
# a while longer. At last it will show both policies in comparison.
# This demonstrates both that training actually improves performace
# as well as the concept of saving a policy.

# The Agent class is simply a way to bundle together a frozen policy
# (not to be furtherly trained) and an environment, so that it can
# simply be ran greedily as a simulation.
class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        
        # These are just to collect some data to compare the agents in the end 
        self.total_score = 0
        self.n_episodes = 0

    def play(self, n=-1, measure=False):
        state = self.env.reset()
        temp_score = 0
        while n != 0:
            action = self.policy(state)
            state, _, done, _ = self.env.step(action)
            self.env.render()
            temp_score += 1
            if done: 
                self.env.reset()
                if measure:
                    self.n_episodes += 1
                    self.total_score += temp_score
                    temp_score = 0
            n -= 1

env = gym.make('CartPole-v1')

spec = NetworkSpecification()
dqn = DQNAgent(env, spec, render=False)

batch_size = 32
num_episodes = 4
training_iter = 500
completion_reward = -10

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing poorly trained agent (A).')
poor_policy = dqn.copy_target_policy()
agent_a = Agent(env, poor_policy)
agent_a.play(500)

print('training...')
dqn.train(num_episodes * 5, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing better agent (B).')
better_policy = dqn.copy_target_policy()
agent_b = Agent(env, better_policy)
agent_b.play(500, measure=True)

print('playing poorly trained agent again (A).')
agent_a.play(500, measure=True)

env.close()
print('Average durations (higher is better): A={}, B={}'.format(agent_a.total_score // agent_a.n_episodes, agent_b.total_score // agent_b.n_episodes))