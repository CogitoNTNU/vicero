# Vicero - Reinforcement Learning Framework <img align="left" width="42" src="https://i.imgur.com/h6uqsjq.png">
This project is supposed to simplify the implementation of a wide range of techniques used for reinforcement learning. At the same time, the techniques themselves should be implemented cleanly and with precise yet easy to understand documentation.
# Setup guide for developers
To setup the project after cloning, run the following command in the project root. You might need to prefix it with sudo, if you don't have root privilegies already.
```
pip install -e .
```
This installs the current directory as if it was any python library, but with the -e (editable) flag it will keep the installation up-to-date with any edit to the source code.

# Dependencies
```
numpy
pytorch
pydot
pygame
matplotlib
gym
```
# Known issues
- The term "agent" is used a bit inconsistently through the project, this is a problem that will be solved over time as the naming conventions solidify.
- TicTacToe, although being one of the example environments, is not handled well by the MCTS implementation at the moment. The concept of a draw being better than a loss needs to be addressed in a more clean way. 
- The NetworkSpecification class should by all means support more layer types than just fully connected ones, but defining a smart high level interface for this where the user doesn't have to import anything from pytorch will be a challenge.

# References
Reinforcement Learning, Sutton and Barto 2018 _(abbreviated in the source as S&B18)_