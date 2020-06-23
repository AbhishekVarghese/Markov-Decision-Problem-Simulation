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
player_pos = (2,3)

test_act = [[1,1,1,5],
			[3,5,1,5],
			[1,1,2,2]]
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

# -0.02 everywhere except 2 places
reward = [ [-0.02]*len(board[0]) for _ in range(len(board)) ]
reward[0][3] = 1
reward[1][3] = -1
reward[1][1] = 0

	
#Action - consequence the actual prob distribution which only environment knows
action_consequence = dict()
action_consequence[0] = [(3,0.1),(0,0.8),(2,0.1)]
action_consequence[1] = [(2,0.1),(1,0.8),(3,0.1)]
action_consequence[2] = [(0,0.1),(2,0.8),(1,0.1)]
action_consequence[3] = [(1,0.1),(3,0.8),(0,0.1)]

#Miscallanepous
tile_size = 100
n_tiles_horiz = len(board[0])
n_tiles_vertical = len(board)
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

def value_iteration():
		global value_estimate,action

		for i in range(len(value_estimate)) :
			for j in range(len(value_estimate[i])) :
				value_estimate[i][j] = 0
		
		for i in range(len(value_estimate)):
			for j in range(len(value_estimate[i])) :
				if action[i][j] == 5 :
					pass
				to_take_max = []
				for each_act in [0,1,2,3] :
					possible_actions = action_consequence_estimates[each_act]
					overall_sum = 0
					for act in possible_actions :
						if act == "ntimes" :
							continue
						if act == action[i][j] :
							prob = 0.8
						else :
							prob = 0.1
						# prob = possible_actions[act]/possible_actions["ntimes"]
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
				value_estimate[i][j] = reward[i][j] + max(to_take_max)
				action[i][j] = np.argmax(to_take_max)


def policy_iteration_onestep(action,value_estimate) :

	A = np.zeros((n_tiles_horiz*n_tiles_vertical,n_tiles_horiz*n_tiles_vertical), dtype="float32")
	for i in range(len(value_estimate)) :
		for j in range(len(value_estimate[i])):
			A[flatten(i,j),flatten(i,j)] = 1
			if action[i][j] == 5 :
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
	
	A = np.delete(np.delete(A,flatten(1,1),0), flatten(1,1),1)
	# print(A.shape, A)
	B = np.delete(np.array(reward).flatten(), flatten(1,1))
	# print(B)
	X = np.insert(np.linalg.inv(A).dot(B), flatten(1,1),0)
	# print(X)
	for i in range(len(value_estimate)) :
		for j in range(len(value_estimate[i])):
			if board[i][j] != 2:
				value_estimate[i][j] = X[flatten(i,j)]

	print("Old Action : ",action)
	for i in range(len(action)):
		for j in range(len(action[i])) :
			if action[i][j] == 5 :
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
					if act == action[i][j] :
						prob = 0.8
					else :
						prob = 0.1
					# prob = possible_actions[act]/possible_actions["ntimes"]
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
	prev_val = np.sum(value_estimate)
	policy_iteration_onestep(action,value_estimate)
	for i in range(500) :
		print(prev_val ,np.sum(value_estimate))
		prev_val = np.sum(value_estimate)
		policy_iteration_onestep(action,value_estimate)


def agent_move() :
	global board,player_pos,action_consequence_estimates
	
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
def click():
	pass

def one_step():
	global player_pos, num_moves, type_move

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
			if board[i][j] in (0,1):
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
	# draw_arrow(canvas,action)
	

	
# Create a frame and assign callbacks to event handlers
# policy_iteration(test_act,value_estimate)
# print(test_act)
# print(value_estimate)
# quit()

frame = simplegui.create_frame("Home", tile_size*n_tiles_horiz, tile_size*n_tiles_vertical)
frame.add_button("1 Step", one_step)
frame.add_button("Single Run", single_run)
frame.add_button("Play", play)
frame.add_button("Stop", stop)
frame.add_button("Reset", click)
frame.set_draw_handler(draw_all)

timer_play = simplegui.create_timer(500, one_step)
timer_singlerun = simplegui.create_timer(500, one_step)

# Start the frame animation
frame.start()
