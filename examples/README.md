# Examples
The examples module contains examples of usage as well as
problem generators / simulators / environments.

## Environments

# Maze
The maze environment is a randomly generated solvable maze.
This environment also entails designed challenges that might
not be formally classified as mazes.
Actions: up, down, left, right
Reward: 0 or -1 for each step, 1 for solving the maze

# Gridworld
Adapted from S&B18, the gridworld is a basic grid without
collisions, where the goal is expressed as one or more 
specific cells.
Similar to maze.

# SMaze
Same as maze, but the environment might change at any time
after some user-specified rules. This is to test the 
adaptiveness of an agent.

# Hex
Hexagon based board game. Each player "own" one pair of 
diagonal sides on a diamond shaped board. In alternating
turns they place pieces of their color. The first player
to bridge across the board wins. The game is proven not
to end in a tie.

# Cartpole
From OpenAI Gym
https://gym.openai.com/envs/CartPole-v1/

# MountainCar
From OpenAI Gym
https://gym.openai.com/envs/MountainCar-v0/
