# Markov-Decision-Problem-Simulation

This is a simulation of the standard textbook MDP environment(Robot in a maze) made using SimpleGUI and python.

To install SimpleGUI 

    pip install SimpleGUICS2Pygame
    

### Environment - Robot in a 2D maze :

**Goal** - Get to the green square without touching the red square within certain number of moves. If you hit a wall(grey square) you will bounce back to the same square.

**States** - Squares on the grid with cordinates (x,y)

**Actions** - Up, Down, Left or Right 

**Rewards** - +1 on reaching the goal, -1 on hitting red square, 0 on all others

**Randomness** - When an agent takes an action it goes in the correct direction with probability 0.8, or with a probability of 0.1 it can steer either to the left or to the right.

The V value and the best action on each square is available.

The boards can be edited within the code to add new or bigger boards. Further instructions are given within the code.


### Files

There are 2 files.

1. MDP.py is the simulation
1. MDP_environment is to understand the basic environment without the randomness.
