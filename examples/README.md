# Examples
The examples module contains examples of usage as well as
problem generators / simulators / environments.
The purpose is both to show the usage of the vicero library
as well as general demonstrations of reinforcement learning.

## Environments

### Maze
The maze environment is a randomly generated solvable maze.
This environment also entails designed challenges that might
not be formally classified as mazes.
Actions: up, down, left, right
Reward: 0 or -1 for each step, 1 for solving the maze

### Gridworld
Adapted from S&B18, the gridworld is a basic grid without
collisions, where the goal is expressed as one or more 
specific cells.
Similar to maze.

### SMaze (Stochastic Maze)
Same as maze, but the environment might change at any time
after some user-specified rules. This is to test the 
adaptiveness of an agent. Some interesting situations
in this environment are highlighted in S&B18.

### Hex
Hexagon based board game. Each player "own" one pair of 
diagonal sides on a diamond shaped board. In alternating
turns they place pieces of their color. The first player
to bridge across the board wins. The game is proven not
to end in a tie.

### m,n,k game
A generalization of games like tic-tac-toe, where
(m, n) are the board dimensions, and k is the number of
pieces a playes has to place in a straight line in order
to win.

### Cartpole
From OpenAI Gym
https://gym.openai.com/envs/CartPole-v1/

### MountainCar
From OpenAI Gym
https://gym.openai.com/envs/MountainCar-v0/

## Visualization

### Pygame
Pygame is a python library for making games, but in this
project it is mostly used for visualizing the environment.
Adding an agent that responds to user input is also
possible, if you want to interact with the AI :)

### Matplotlib
More traditional plotting is typically more useful than a
simulation when you are interested in the performance rather 
than the policy behaviour. Typically plotted values are 
return (cumulative reward from each episode), neural net
accuracy, confidence etc.