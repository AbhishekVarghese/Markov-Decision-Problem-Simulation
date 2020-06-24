# CodeSkulptor runs Python programs in your browser.
# Click the upper left button to run this simple demo.

# CodeSkulptor is tested to run in recent versions of
# Chrome, Firefox, and Safari.

# Markov Decision Process Simulation
# Created by Abhishek Varghese


import random, time
from bisect import bisect
import numpy as np
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 



seed = 13
random.seed(seed)


#Building the environment

# Board Values
# 0 is empty
# 1 is None right now
# 2 is wall
# 3 is hollow
# 4 is goal
board1 = [[0,0,0,4],
		 [0,2,0,3],
		 [0,0,0,0]]
start_pos1 = (2,3)

board2= [[0,0,4,0,0],
		 [2,0,2,0,3],
		 [0,0,0,3,0],
		 [0,2,0,0,0]]
start_pos2 = (3,4)



board = board1
start_pos = start_pos1

player_pos = start_pos


# 0 is left
# 1 is right
# 2 is up
# 3 is down
# None is No action available
action = [ [0]*len(board[0]) for _ in range(len(board)) ]
print(action)
for i in range(len(action)) :
	for j in range(len(action[i])) :
		if board[i][j] == 0 :
			action[i][j] = random.choice([0,1,2,3])
		else :
			action[i][j] = None

# action = test_act

# +1 for goal, -1 for danger and -0.02 everywhere
reward = [ [-0.02]*len(board[0]) for _ in range(len(board)) ]
for i in range(len(reward)):
	for j in range(len(reward[i])) :
		if board[i][j] == 4 :
			reward[i][j] = 1
		elif board[i][j]== 3 :
			reward[i][j] = -1
		else :
			reward[i][j] = -0.02

	
#Action - consequence the actual prob distribution which only environment knows
action_consequence = dict()
action_consequence[0] = [(3,0.1),(0,0.8),(2,0.1)]
action_consequence[1] = [(2,0.1),(1,0.8),(3,0.1)]
action_consequence[2] = [(0,0.1),(2,0.8),(1,0.1)]
action_consequence[3] = [(1,0.1),(3,0.8),(0,0.1)]

#Miscallanepous
canvas_width = 700
canvas_height = 500
n_tiles_horiz = len(board[0])
n_tiles_vertical = len(board)
tile_size = canvas_height/n_tiles_vertical

tile_dims = (tile_size,tile_size)
arrow_size = tile_size/6
num_moves = 0
type_move = "player"

#---------------------------------------------------------------------------------------------

#Following are the parts of the agent's brain. Only to be used in estimate functions

gamma = 0.99
#Action - consequence estimates [action,# of times happened, #of times taken]
action_consequence_estimates = dict()
action_consequence_estimates[0] = {"ntimes": 3, 3 : 1,0 : 1,2 : 1}
action_consequence_estimates[1] = {"ntimes": 3, 2 : 1,1 : 1,3 : 1}
action_consequence_estimates[2] = {"ntimes": 3, 0 : 1,2 : 1,1 : 1}
action_consequence_estimates[3] = {"ntimes": 3, 1 : 1,3 : 1,0 : 1}

#Initialising the Values
value_estimate = [ [0]*len(board[0]) for _ in range(len(board)) ]
value_estimate = np.array(value_estimate, dtype="float32")

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

def policy_iteration_onestep(action,value_estimate) :

	A = np.zeros((n_tiles_horiz*n_tiles_vertical,n_tiles_horiz*n_tiles_vertical), dtype="float32")
	for i in range(len(value_estimate)) :
		for j in range(len(value_estimate[i])):
			A[flatten(i,j),flatten(i,j)] = 1
			if action[i][j] == None :
				continue
			possible_actions = action_consequence_estimates[action[i][j]]
			for act in possible_actions :
				if act == "ntimes" :
					continue
				
				# if act == action[i][j] :
				# 	prob = 0.8
				# else :
				# 	prob = 0.1
				prob = possible_actions[act]/possible_actions["ntimes"]
				if act == 0 :
					if  j != 0 and board[i][j-1] != 2:
						A[flatten(i,j),flatten(i,j-1)] += -prob*gamma
					else :
						A[flatten(i,j),flatten(i,j)] += -prob*gamma
				elif act == 1 :
					if j != len(value_estimate[i]) - 1 and board[i][j+1] != 2: 
						A[flatten(i,j),flatten(i,j+1)] += -prob*gamma
					else :
						A[flatten(i,j),flatten(i,j)] += -prob*gamma
				elif act == 2 :
					if i != 0  and board[i-1][j] != 2:
						A[flatten(i,j),flatten(i-1,j)] += -prob*gamma
					else :
						A[flatten(i,j),flatten(i,j)] += -prob*gamma
				elif act == 3 :
					if i != len(value_estimate) - 1  and board[i+1][j] != 2:
						A[flatten(i,j),flatten(i+1,j)] += -prob*gamma
					else :
						A[flatten(i,j),flatten(i,j)] += -prob*gamma
	
	# A = np.delete(np.delete(A,flatten(1,1),0), flatten(1,1),1)
	# print(A.shape, A)
	# B = np.delete(np.array(reward).flatten(), flatten(1,1))
	B = np.array(reward).flatten()
	# print(B)
	# X = np.insert(np.linalg.inv(A).dot(B), flatten(1,1),0)
	X = np.linalg.inv(A).dot(B)
	# print(X)
	for i in range(len(value_estimate)) :
		for j in range(len(value_estimate[i])):
			if board[i][j] != 2:
				value_estimate[i][j] = X[flatten(i,j)]

	print("Old Action : ",action)
	for i in range(len(action)):
		for j in range(len(action[i])) :
			if action[i][j] == None :
				continue
			to_take_max = []
			for each_act in [0,1,2,3] :
				if (each_act == 0 and  (j == 0 or board[i][j-1]==2)) or (each_act==1 and (j == len(value_estimate[i]) - 1 or board[i][j+1]==2 )) or (each_act==2 and (i == 0 or board[i-1][j]==2)) or (each_act ==3 and (i == len(value_estimate) - 1 or board[i+1][j]==2)) :
					to_take_max.append(-20)
					continue
				possible_actions = action_consequence_estimates[each_act]
				overall_sum = 0
				for act in possible_actions :
					if act == "ntimes" :
						continue
					# if act == action[i][j] :
					# 	prob = 0.8
					# else :
					# 	prob = 0.1
					prob = possible_actions[act]/possible_actions["ntimes"]
					if act == 0 :
						if  j != 0 and board[i][j-1] != 2:
							overall_sum += prob*value_estimate[i][j-1]
						else :
							overall_sum += prob*value_estimate[i][j]
					elif act == 1 :
						if j != len(value_estimate[i]) - 1 and board[i][j+1] != 2: 
							overall_sum += prob*value_estimate[i][j+1]
						else :
							overall_sum += prob*value_estimate[i][j]
					elif act == 2 :
						if i != 0  and board[i-1][j] != 2:
							overall_sum += prob*value_estimate[i-1][j]
						else :
							overall_sum += prob*value_estimate[i][j]
					elif act == 3 :
						if i != len(value_estimate) - 1  and board[i+1][j] != 2:
							overall_sum += prob*value_estimate[i+1][j]
						else :
							overall_sum += prob*value_estimate[i][j]
				to_take_max.append(overall_sum)
			
			print("i : ",i,"j : ",j, "list = ",to_take_max)
			action[i][j] = np.argmax(to_take_max)
			# action[i][j] = np.argmax([  value_estimate[i][max(0,j-1)], value_estimate[i][min(n_tiles_horiz -1,j+1)],
			# 		value_estimate[max(i-1,0)][j], value_estimate[min(i+1,n_tiles_vertical-1)][j] ])
	print("New Action List : ",action)

def policy_iteration(action,value_estimate) :
	for i in range(100) :
		policy_iteration_onestep(action,value_estimate)


def agent_move() :
	global board,player_pos,action_consequence_estimates,action

	print(len(action), len(action[1]))	
	sampled_action = weighted_choice(action_consequence[action[player_pos[0]][player_pos[1]]])
	# Agent Updating what has happened in his mind
	action_consequence_estimates[action[player_pos[0]][player_pos[1]]][sampled_action] += 1
	action_consequence_estimates[action[player_pos[0]][player_pos[1]]]["ntimes"] += 1

	
	if sampled_action == 0 :
		new_pos = (player_pos[0],player_pos[1] -1)
		if new_pos[1] < 0 or board[new_pos[0]][new_pos[1]] == 2:
			pass
		else :
			player_pos = new_pos
	elif sampled_action == 1 :
		new_pos = (player_pos[0],player_pos[1] + 1)
		if new_pos[1] >= n_tiles_horiz or board[new_pos[0]][new_pos[1]] == 2:
			pass
		else :
			player_pos = new_pos
	elif sampled_action == 2 :
		new_pos = (player_pos[0]-1,player_pos[1])
		if new_pos[0] < 0 or board[new_pos[0]][new_pos[1]] == 2:
			pass
		else :
			player_pos = new_pos
	elif sampled_action == 3 :
		new_pos = (player_pos[0] + 1,player_pos[1])
		if new_pos[0] >= n_tiles_vertical or board[new_pos[0]][new_pos[1]] == 2:
			pass
		else :
			player_pos = new_pos
	
	if reward[player_pos[0]][player_pos[1]] in (1,-1) :
		return False

	return True
# -------------------------------------------------------------------------------------

# Handler for mouse click
def set_board1() :
	global board,start_pos

	board = board1 
	start_pos = start_pos1
	reset()

def set_board2() :
	global board,start_pos

	board = board2 
	start_pos = start_pos2
	reset()

def reset():
	global player_pos,action_consequence_estimates,value_estimate, seed,reward, board,action,n_tiles_horiz,n_tiles_vertical,tile_size,arrow_size,label1,label2,label3


	player_pos = start_pos
	seed += 1
	random.seed(seed)
	#Reser action consequence estimates
	action_consequence_estimates = dict()
	action_consequence_estimates[0] = {"ntimes": 3, 3 : 1,0 : 1,2 : 1}
	action_consequence_estimates[1] = {"ntimes": 3, 2 : 1,1 : 1,3 : 1}
	action_consequence_estimates[2] = {"ntimes": 3, 0 : 1,2 : 1,1 : 1}
	action_consequence_estimates[3] = {"ntimes": 3, 1 : 1,3 : 1,0 : 1}
	label1.set_text('P_left_goes-left = %.3f'%(action_consequence_estimates[0][0]/action_consequence_estimates[0]["ntimes"]))
	label2.set_text('P_left_goes-up = %.3f'%(action_consequence_estimates[0][2]/action_consequence_estimates[0]["ntimes"]))
	label3.set_text('P_left_goes-down = %.3f'%(action_consequence_estimates[0][3]/action_consequence_estimates[0]["ntimes"]))


	#reset value_estimates
	value_estimate = [ [0]*len(board[0]) for _ in range(len(board)) ]
	value_estimate = np.array(value_estimate, dtype="float32")

	#reset action to random
	action = [ [0]*len(board[0]) for _ in range(len(board)) ]
	for i in range(len(action)) :
		for j in range(len(action[i])) :
			if board[i][j] == 0 :
				action[i][j] = random.choice([0,1,2,3])
			else :
				action[i][j] = None

	#Stop all timers
	timer_play.stop()
	timer_singlerun.stop()

	#reset reward
	reward = [ [-0.02]*len(board[0]) for _ in range(len(board)) ]
	for i in range(len(reward)):
		for j in range(len(reward[i])) :
			if board[i][j] == 4 :
				reward[i][j] = 1
			elif board[i][j]== 3 :
				reward[i][j] = -1
			else :
				reward[i][j] = -0.02

	n_tiles_horiz = len(board[0])
	n_tiles_vertical = len(board)
	tile_size = canvas_height/n_tiles_vertical

	tile_dims = (tile_size,tile_size)
	arrow_size = tile_size/6


def one_step():
	global player_pos, num_moves, type_move, label1,label2,label3

	num_moves += 1
	if type_move == "player" :
		wandering = agent_move()
		if not wandering or num_moves > 10:
			type_move = "computation"
	elif type_move == "computation" :
		if timer_play.is_running() :
			timer_play.stop()
			category = "play"
		elif timer_singlerun.is_running():
			timer_singlerun.stop()
			category = "singlerun"
		timer_singlerun.stop()
		label1.set_text('P_left_goes-left = %.3f'%(action_consequence_estimates[0][0]/action_consequence_estimates[0]["ntimes"]))
		label2.set_text('P_left_goes-up = %.3f'%(action_consequence_estimates[0][2]/action_consequence_estimates[0]["ntimes"]))
		label3.set_text('P_left_goes-down = %.3f'%(action_consequence_estimates[0][3]/action_consequence_estimates[0]["ntimes"]))
		policy_iteration(action, value_estimate)
		type_move = "blank"
		if category=="play" :
			timer_play.start()
		elif category == "singlerun" :
			timer_singlerun.start()
		
	else :
		type_move= "player"
		player_pos = start_pos
		num_moves = 0
		timer_singlerun.stop()

def single_run() :
	global timer_singlerun
	timer_play.stop()
	timer_singlerun.start()

def play() :
	global timer_play
	timer_singlerun.stop()
	timer_play.start()

def stop():
	global timer_play
	timer_singlerun.stop()
	timer_play.stop()
# -----------------------------------------------------------------------------------------------

# Handler to draw on canvas

def draw_arrow(canvas,action):
	padding = tile_size/12
	inner_tile_size = tile_size - 2*padding
	for i in range(len(action)) :
		curr_pos_vertical = tile_size*i + padding
		for j in range(len(action[i])):
			curr_pos_horizontal = tile_size*j + padding
			canvas.draw_text(str(value_estimate[i][j]), (curr_pos_horizontal , curr_pos_vertical + 1*padding ), 10, 'white')
			if action[i][j] != None:
				if action[i][j] == 2 :
					point_list =[[curr_pos_horizontal,curr_pos_vertical + inner_tile_size],
								 [curr_pos_horizontal + arrow_size/2,curr_pos_vertical + inner_tile_size - arrow_size],
								 [curr_pos_horizontal + arrow_size,curr_pos_vertical + inner_tile_size]]
				elif action[i][j] == 0 :
					point_list =[[curr_pos_horizontal,curr_pos_vertical + inner_tile_size - arrow_size/2],
								 [curr_pos_horizontal + arrow_size,curr_pos_vertical + inner_tile_size - arrow_size],
								 [curr_pos_horizontal + arrow_size,curr_pos_vertical + inner_tile_size]]
				elif action[i][j] == 1 :
					point_list =[[curr_pos_horizontal,curr_pos_vertical + inner_tile_size- arrow_size],
								 [curr_pos_horizontal,curr_pos_vertical + inner_tile_size ],
								 [curr_pos_horizontal + arrow_size,curr_pos_vertical + inner_tile_size - arrow_size/2]]
				elif action[i][j] == 3 :
					point_list =[[curr_pos_horizontal,curr_pos_vertical + inner_tile_size-arrow_size],
								 [curr_pos_horizontal + arrow_size,curr_pos_vertical + inner_tile_size - arrow_size],
								 [curr_pos_horizontal + arrow_size/2,curr_pos_vertical + inner_tile_size]]		
				canvas.draw_polygon(point_list, 1, "white", "white")

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
	draw_arrow(canvas,action)
	

	
# Create a frame and assign callbacks to event handlers
# policy_iteration(test_act,value_estimate)
# print(test_act)
# print(value_estimate)
# quit()

frame = simplegui.create_frame("Home", canvas_width, canvas_height,400)
frame.add_button("1 Step", one_step)
frame.add_button("Single Run", single_run)
frame.add_button("Play", play)
frame.add_button("Stop", stop)
frame.add_button("Reset", reset)
frame.set_draw_handler(draw_all)

print(action_consequence_estimates)
frame.add_label("\n\n\n")
frame.add_label("\n\n\n")
frame.add_label("\n\nProbability Distribution known by the agent for action left : \n\n")
frame.add_label("\n\n\n")
label1 = frame.add_label('P_left_goes-left = %.3f'%(action_consequence_estimates[0][0]/action_consequence_estimates[0]["ntimes"]))
label2 = frame.add_label('P_left_goes-up = %.3f'%(action_consequence_estimates[0][2]/action_consequence_estimates[0]["ntimes"]))
label3 =  frame.add_label('P_left_goes-down = %.3f'%(action_consequence_estimates[0][3]/action_consequence_estimates[0]["ntimes"]))
frame.add_label("\n\n\n")
frame.add_label("\n\n\n")
frame.add_button("Board1", set_board1)
frame.add_button("Board2", set_board2)


timer_play = simplegui.create_timer(500, one_step)
timer_singlerun = simplegui.create_timer(500, one_step)

# Start the frame animation
frame.start()
