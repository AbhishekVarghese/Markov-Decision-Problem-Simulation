# CodeSkulptor runs Python programs in your browser.
# Click the upper left button to run this simple demo.

# CodeSkulptor is tested to run in recent versions of
# Chrome, Firefox, and Safari.

# Markov Decision Process Simulation
# Created by Abhishek Varghese


import random, time
from bisect import bisect
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 


random.seed(11)
#Building the environment

# Board Values
# 0 is empty
# 1 is None right now
# 2 is wall
# 3 is hollow
# 4 is goal
board = [[0,0,0,4],
         [0,2,0,3],
         [0,0,0,0]]
start_pos = (2,3)
player_pos = [2,3]


# 0 is left
# 1 is right
# 2 is up
# 3 is down
# 5 is No action available
action = [ [0]*len(board[0]) for _ in range(len(board)) ]
print(action)
for i in range(len(action)) :
    for j in range(len(action[i])) :
        action[i][j] = random.choice([0,1,2,3])
action[0][3] = 5
action[1][3] = 5
action[1][1] = 5
# action = test_act



#Miscallanepous
tile_size = 100
n_tiles_horiz = len(board[0])
n_tiles_vertical = len(board)
tile_dims = (tile_size,tile_size)
arrow_size = tile_size/6
num_moves = 0
type_move = "player"

# ----------------------------------------------------------------------------------------------


#Helper Functions
def weighted_choice(data):
    values,weights = zip(*data)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.uniform(0,1)
    i = bisect(cum_weights, x)
    return values[i]

def flatten(i,j) :
    return n_tiles_horiz*i + j



# -------------------------------------------------------------------------------------

# Handler for mouse click
def left():
    global player_pos
    if player_pos[1] != 0 :
        player_pos[1] -= 1
        

def right():
    global player_pos
    
    if player_pos[1] !=  n_tiles_horiz - 1:
        player_pos[1] += 1
        
def up():
    global player_pos
    if player_pos[1] == 2 and player_pos[0] == 1 :
        player_pos[1] += 1
        return
    
    if player_pos[0] != 0 :
        player_pos[0] -= 1
        
def down():
    global player_pos
    if player_pos[0] != n_tiles_vertical - 1 :
        player_pos[0] += 1
# -----------------------------------------------------------------------------------------------

# Handler to draw on canvas

def draw_board(canvas, board) :
    for i in range(len(board)) :
        curr_pos_vertical = tile_size*i
        for j in range(len(board[i])):
            fill_color = "black"
            if board[i][j] == 2 :
                fill_color = "grey"
            if board[i][j] == 3 :
                fill_color = "red"
            if board[i][j] == 4 :
                fill_color = "green"
            
            curr_pos_horizontal = tile_size*j
            point_list = [[curr_pos_horizontal,curr_pos_vertical],[curr_pos_horizontal+tile_size,curr_pos_vertical],
                          [curr_pos_horizontal+tile_size,curr_pos_vertical+tile_size],[curr_pos_horizontal,curr_pos_vertical+tile_size]]
            canvas.draw_polygon(point_list, 10, "blue", fill_color)
            
    canvas.draw_circle((player_pos[1]*tile_size+tile_size/2,player_pos[0]*tile_size+tile_size/2), tile_size/4, 2, "yellow", "yellow")

            
def draw_all(canvas):
    draw_board(canvas,board)
    #draw_arrow(canvas,action)
    

    
# Create a frame and assign callbacks to event handlers
# policy_iteration(test_act,value_estimate)
# print(test_act)
# print(value_estimate)
# quit()

frame = simplegui.create_frame("Home", tile_size*n_tiles_horiz, tile_size*n_tiles_vertical)
frame.add_button("up", up)
frame.add_button("down", down)
frame.add_button("left", left)
frame.add_button("right", right)
frame.set_draw_handler(draw_all)

# Start the frame animation
frame.start()
