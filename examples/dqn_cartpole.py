import gym
from vicero.algorithms.deepqlearning import DQN, NetworkSpecification
from vicero.agent import Agent

# This example is showing off multiple concepts, for a more pure
# DQN example, see mountaincar. The first part is obviously to
# solve cartpole, but in addition, the script will save the policy
# as it is after one shorter round of training, then it will train
# a while longer. At last it will show both policies in comparison.
# This demonstrates both that training actually improves performace
# as well as the concept of saving a policy.

env = gym.make('CartPole-v1')

spec = NetworkSpecification()
dqn = DQN(env, spec, render=False)

batch_size = 32
num_episodes = 4
training_iter = 500
completion_reward = -10

print('training...')
dqn.train(num_episodes, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing poorly trained agent (A).')
poor_policy = dqn.copy_target_policy()
agent_a = Agent(env, poor_policy)
for _ in range(500):
    agent_a.step(render=True)

print('training...')
dqn.train(num_episodes * 10, batch_size, training_iter, verbose=True, completion_reward=completion_reward, plot=True, eps_decay=True)

print('playing better agent (B).')
better_policy = dqn.copy_target_policy()
agent_b = Agent(env, better_policy)
for _ in range(500):
    agent_b.step(render=True, measure=True)

print('playing poorly trained agent again (A).')
for _ in range(500):
    agent_a.step(render=True, measure=True)

env.close()
print('Agent performance (average duration): A={:.1f}, B={:.1f}'.format(agent_a.performance, agent_b.performance))