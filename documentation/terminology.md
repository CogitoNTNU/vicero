## Terminology
# Algorithm
In this context, and algorithm is one of the many
strategies to solve the RL problem.

# Agent
And agent is as it is defined by Russel and Norvig.
More contextually accurate, it is an instance of an
algorithm with a specific set of parameters.
Example: A gamma=0.1 QL agents is a different agent
than a gamma=0.2, but the algorithm is the same.

# Environment
An environment is a generalized concept in this usage.
It includes simulators, interfaces and other problem
wrappers. The requirement for a RL enviroment is that
it can at all times supply a reward signal, perform
state transfer and represent the state. The state 
representation can be furtherly processed before being
fed into the RL algorithm.

# Policy
A policy is a function that takes as input a state, and
produces an action as an output. Producing an optimal 
policy is the primary goal of a RL algorithm.