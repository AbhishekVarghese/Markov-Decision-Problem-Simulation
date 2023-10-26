# Markov-Decision-Problem-Simulation

This is a simulation of the standard textbook MDP environment(Robot in a maze) made using SimpleGUI and python.

To install SimpleGUI 
~~~
pip install SimpleGUICS2Pygame
~~~ 
(You may need to use pip3 instead of pip)

To run the GUI:
~~~
python3 main.py
~~~ 

### Environment - Robot in a 2D maze :

**Goal** - Get to green squares marked with "done state" while avoiding touching the red squares. If you hit a wall (grey square) you will bounce back to the same square.

**States** - Squares on the grid with cordinates (x,y)

**Actions** - Up, Down, Left or Right 

**Rewards** - 
* +1 on reaching green square
* -1 on reaching red square
* 0 on all others

You can change the grid and draw reward states (+1, -1, wall) and mark which states are "done states" by drawing on the canvas. 

After choosing input, you can run either Value Iteration or Q-Learning. 
