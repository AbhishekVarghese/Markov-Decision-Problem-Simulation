import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI,Board
import math
import time
from pygame import Color


def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Agent :
    def __init__(self, state_space_shape, gamma = 0.9,  alpha = 0.1, epsilon = 1) :
        self.alpha = alpha #learning rate
        self.gamma = gamma
        self.state_space_shape = state_space_shape
        self.epsilon = epsilon
        self.action_to_index = {'left' : 0, 'right' : 1, 'up' : 2, 'down' :3}
        self.index_to_action = {0 : 'left', 1: 'right', 2 : 'up', 3: 'down'}
        self.reset_agent()

    def reset_agent(self) :
        # self.estQ = np.random.random(size = self.state_space_shape + (4,) ) - 0.5 # The last indice stores action
        self.estQ = np.zeros(shape = self.state_space_shape + (4,)  )
        self.opt_policy = np.zeros(shape = self.state_space_shape)
        self.estV = np.zeros(shape = self.state_space_shape)

    @property
    def board(self) :
        return self.estQ
    
    def select_action( self, state, available_actions) :
        toss = random.uniform(0,1)
        qvalues = []
        for action in available_actions :
            action_idx = self.action_to_index[action]
            qvalues.append(self.estQ[state[0], state[1], action_idx])

        if toss < self.epsilon :
            # We would like to explore more with higher value of epsilon
            return np.random.choice(available_actions)
        else :
            return available_actions[np.argmax(qvalues)]
    

    def update_state(self, state, action, new_state, reward, legal_actions) :
        # Q = (1 - alpha)*Q(st,at) + alpha(rt + gamma * max_a Q(st+1, a))
        action = self.action_to_index[action]
        Qlhs = (1 - self.alpha)* self.estQ[ state[0], state[1],action]
        Qrhs = self.alpha * ( reward + self.gamma * np.max(self.estQ[ new_state[0], new_state[1], : ]))
        self.estQ[state[0], state[1],action] = Qlhs + Qrhs
        self.opt_policy[state[0], state[1]] = np.argmax(self.estQ[state[0], state[1], :])

        #Compute accurate V_epsilon
        legal_actions_idx = [self.action_to_index[i] for i in legal_actions]
        curr_state_Qvalues = self.estQ[state[0],state[1],legal_actions_idx]
        self.estV[state[0],state[1]] = (1-self.epsilon)*np.max(curr_state_Qvalues) + self.epsilon*np.sum(curr_state_Qvalues)/len(legal_actions_idx)

    

class Mdp_env :
    def __init__(self, board, start_pos = None) :
        self.board_height = board.height
        self.board_width = board.width
        self.reward_dict = board.reward_dict
        self.reward_tiles = board.reward_dict.keys()
        self.blocked_tiles = board.blocked_tiles
        self.done_tiles = board.done_tiles
        self.board = board.board
        self.start_pos = start_pos
        if start_pos :
            self.curr_pos = start_pos
        else :
            all_tiles = set(itertools.product(range(self.board_height, self.board_width)))
            available_tiles = (all_tiles - set(self.reward_tiles)) - set(self.blocked_tiles)
            self.curr_pos = np.random.choice(available_tiles)
        self.visited_tiles = [self.curr_pos]
    
    def reset(self) :
        if self.start_pos :
            self.curr_pos = self.start_pos
        else :
            all_tiles = set(itertools.product(range(self.board_height, self.board_width)))
            available_tiles = (all_tiles - set(self.reward_tiles)) - set(self.blocked_tiles)
            self.curr_pos = np.random.choice(available_tiles)
        self.visited_tiles = [self.curr_pos]
        

    def step(self, action) :
        done = False
        # This piece of code, although repetative, would save memory and speed for huge board dimensions
        if action == 'left' :
            new_pos = (self.curr_pos[0], self.curr_pos[1] - 1)
            if self.curr_pos[1] >= 0 and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go left from this position {self.curr_pos}")
        elif action == 'right' :
            new_pos = (self.curr_pos[0], self.curr_pos[1] + 1)
            if self.curr_pos[1] < self.board_width and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go right from this position {self.curr_pos}")
        elif action == 'up' :
            new_pos = (self.curr_pos[0] - 1, self.curr_pos[1])
            if self.curr_pos[0] >= 0 and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go up from this position {self.curr_pos}")
        elif action == 'down' :
            new_pos = (self.curr_pos[0] + 1 , self.curr_pos[1])
            if self.curr_pos[0] < self.board_height and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go down from this position {self.curr_pos}")
        
        reward  = self.reward_dict[self.curr_pos]
        if self.curr_pos in self.done_tiles :
            done = True
        self.visited_tiles.append(self.curr_pos)

        return self.curr_pos, reward, done

    def get_legal_actions(self) :
        legal_actions = []
        preferred_actions = []
        if self.curr_pos[1] >= 1 and not (self.curr_pos[0], self.curr_pos[1] - 1) in self.blocked_tiles :
            legal_actions.append('left')
            if not (self.curr_pos[0], self.curr_pos[1] - 1) in self.visited_tiles :
                preferred_actions.append('left')
        if self.curr_pos[1] < self.board_width - 1 and not (self.curr_pos[0] , self.curr_pos[1] + 1) in self.blocked_tiles :
            legal_actions.append('right')
            if not (self.curr_pos[0] , self.curr_pos[1] + 1) in self.visited_tiles :
                preferred_actions.append('right')
        if self.curr_pos[0] >= 1 and not (self.curr_pos[0] - 1, self.curr_pos[1]) in self.blocked_tiles :
            legal_actions.append('up')
            if not (self.curr_pos[0] - 1, self.curr_pos[1]) in self.visited_tiles :
                preferred_actions.append('up')
        if self.curr_pos[0] < self.board_height - 1 and not (self.curr_pos[0] + 1, self.curr_pos[1]) in self.blocked_tiles :
            legal_actions.append('down')
            if not (self.curr_pos[0] + 1, self.curr_pos[1]) in self.visited_tiles :
                preferred_actions.append('down')
        
        return legal_actions,preferred_actions


class Qlearning_with_GUI() :
    def __init__(self, frame, gamma = 0.9, epsilon = 1, alpha= 1) :
        self.frame = frame
        self.epsilon = epsilon
        self.avoid_visited_states = False
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_T_in_str = ""
        self.just_done = False
        self.show_Vepsilon_text = True
        self.show_Qval = True
        self.draw_policy = False
        


        #Prevent Concurrency issues i.e. Functions are called faster than they can finish using the timer
        self.single_step_running = False

    def take_over(self, board, start_pos) :
        self.board = board
        print(board.board)
        self.env = Mdp_env(board,start_pos=start_pos)
        self.agent = Agent((self.env.board_height, self.env.board_width), self.gamma, self.alpha, self.epsilon)
        self.start_pos = start_pos
        self.curr_T = 0
        self.setup_frame()
        self.set_pad_l()

        # For releasing control
        self.player_pos = start_pos

    def set_control_transfer(self, send_control_to):
        self.send_control_to = send_control_to

    def release_control(self, target):
        def handler():
            if not hasattr(self, "send_control_to"):
                return 
            if type(self.send_control_to) != dict or target not in self.send_control_to:
                send_fn = self.send_control_to
            else:
                send_fn = self.send_control_to[target]
            if hasattr(self, "timer_play") and self.timer_play.is_running:
                self.timer_play.stop()
            self.draw_mode = None
            self.timer_play.stop()
            self.frame._controls = []
            self.frame._draw_controlpanel()
            send_fn(
                self.board, self.player_pos, 
            )
        return handler
        
    def setup_frame(self) :
        # Board visualisation constants
        self.background_color = np.array([248, 248, 255]) 
        self.grid_color = "black"
        self.text_color = "black"
        self.arrow_color = "black"
        self.wall_color = "grey"
        self.agent_color = (255, 215, 0)
        self.grid_width = 2
        self.text_size = 12
        self.draw_mode = 0
        self.cmap_negvval = np.array([199, 21, 133])
        self.cmap_posvval = np.array([0, 250, 154])
        self.cmap_neg_to_pos = self.cmap_posvval - self.cmap_negvval
        self.cmap_bag_to_pos = self.cmap_posvval - self.background_color
        self.cmap_bag_to_neg = self.cmap_negvval - self.background_color
        self.default_timer_speed = 1000
        self.speed_mod_factor = 1
        self.speed_increment = 0.1
        self.cell_pad_ratio = 0.25

        self.cmap = {
            0: f"rgb{tuple(self.background_color)}",
            1: f"rgb{tuple(self.cmap_posvval)}",
            -1: f"rgb{tuple(self.cmap_negvval)}",
            -10: self.wall_color,
            "other": "cyan"
        }
        self.arrow_width = 0.02
        w = self.arrow_width
        self.arrow_polygons = {
            "right": [
                [0.5 - w/2, 0.1],
                [0.5 + w/2, 0.1],
                [0.5 + w/2, 0.8],
                [0.7, 0.7],
                [0.5, 0.9],
                [0.3, 0.7],
                [0.5 - w/2, 0.8],
            ],
        }
        self.arrow_polygons["left"] = [
            (i, 1-j) for i, j in self.arrow_polygons["right"]
        ]
        self.arrow_polygons["down"] = [
            (j, i) for i, j in self.arrow_polygons["right"]
        ]
        self.arrow_polygons["up"] = [
            (1-j, i) for i, j in self.arrow_polygons["right"]
        ]
        

        # UI parameter imports
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height

        #Setting up timers
        self.timer_play = simplegui.create_timer(self.default_timer_speed, self.single_step)

        #Set up background color
        self.frame._canvas._background_pygame_color = Color(self.background_color)

        # Adding GUI elements
        self.gamma_label = self.frame.add_input("Discount Factor (Gamma)", self.update_gamma, width=100)
        self.gamma_label.set_text(str(self.gamma))
        self.epsilon_label = self.frame.add_input("Exploration Rate (Epsilon)", self.update_epsilon, width=100)
        self.epsilon_label.set_text(str(self.epsilon))
        self.alpha_label = self.frame.add_input("Learning Rate (Alpha)", self.update_alpha, width=100)
        self.alpha_label.set_text(str(self.alpha))
        self.T_label = self.frame.add_input("Max time steps (T)", self.update_T, width=100)
        self.T_label.set_text(self.max_T_in_str)
        self.frame.add_label("\n"*6)
        self.frame.add_button("Run Sim", self.run_sim)
        self.frame.add_button("Stop Sim", self.stop_sim)
        self.frame.add_button("Reset", self.reset)

        self.button_avoid_visited_states = self.frame.add_button(f"Try avoid visited states in this episode : {self.avoid_visited_states}", self.flip_avoid_states)
        self.frame.add_label("\n"*6)
        self.speed_label = self.frame.add_input("Animation Speed multiplier", self.update_speed("set"), width=100)
        self.speed_label.set_text(str(self.speed_mod_factor))
        # self.frame.add_button("+", self.update_speed("+"), width = 100)
        # self.frame.add_button("--", self.update_speed("-"), width = 100)
        self.animation_freq_display_label = self.frame.add_label(f"Taking action every {self.timer_play._interval*100//100/1000}s")
        
        self.frame.set_draw_handler(self.draw_board)
        self.frame.add_label("\n"*10)

        self.frame.add_button("Back to input", self.release_control("input"), width = 200)
        self.frame.add_button("Switch to Value Iteration", self.release_control("value_iteration"), width = 200)
        

        self.frame.add_label("\n"*10)
        self.frame.add_label("Hide/Show more details. May affect the FPS of the display.")
        self.fps_label = self.frame.add_label("FPS : 0")
        self.show_text_button = self.frame.add_button(f"Show V_epsilon in text : {self.show_Vepsilon_text}", self.flip_show_text)
        self.show_Qvalues_button = self.frame.add_button(f"Show Qvalues as ColorMap : {self.show_Qval}", self.flip_show_qval)
        self.show_policy_button = self.frame.add_button(f"Show Optimal Policy argmax(Q(S)) : {self.draw_policy}", self.flip_show_policy)

    def flip_avoid_states(self) :
        self.avoid_visited_states = not self.avoid_visited_states
        self.button_avoid_visited_states.set_text(f"Try avoid visited states in this episode : {self.avoid_visited_states}")
    
    def flip_show_qval(self) :
        self.show_Qval = not self.show_Qval
        self.show_Qvalues_button.set_text(f"Show Qvalues as ColorMap : {self.show_Qval}")

    def flip_show_text(self) :
        self.show_Vepsilon_text = not self.show_Vepsilon_text
        self.show_text_button.set_text(f"Show V_epsilon in text : {self.show_Vepsilon_text}")

    def flip_show_policy(self) :
        self.draw_policy = not self.draw_policy
        self.show_policy_button.set_text(f"Show Optimal Policy argmax(Q(S)) : {self.draw_policy}")

    def update_T(self,T) :
        if T.isdigit() :
            self.max_T_in_str = str(int(T))
            print("Changed T to", self.max_T_in_str)
        else :
            self.max_T_in_str = ""
        self.T_label.set_text(self.max_T_in_str)


    def update_gamma(self,gamma) :
        if gamma.replace('.', '', 1).isdigit() :
            gamma = float(gamma)
            if  0 <= gamma <= 1 :
                self.gamma = gamma
                self.agent.gamma = gamma
            print("Changed gamma to", self.gamma)
        self.gamma_label.set_text(str(self.gamma))

    def update_epsilon(self,epsilon) :
        if epsilon.replace('.', '', 1).isdigit() :
            epsilon = float(epsilon)
            if  0 <= epsilon <= 1 :
                self.epsilon = epsilon
                self.agent.epsilon = epsilon
            print("Changed epsilon to", self.epsilon)
        self.epsilon_label.set_text(str(self.epsilon))

    def update_alpha(self,alpha) :
        if alpha.replace('.', '', 1).isdigit() :
            alpha = float(alpha)
            if 0 <= alpha :
                self.alpha = alpha
                self.agent.alpha = alpha
            print("Changed alpha to", alpha)
        self.alpha_label.set_text(str(self.alpha))

    def update_speed(self, mode):
        def common(x):
            assert (x > 0), "Error: Animation Frequency Modifier has to be >=0"
            self.speed_mod_factor = x
            self.speed_label.set_text("{:.3f}".format(self.speed_mod_factor))
            print("Setting Animation time interval to", self.default_timer_speed/self.speed_mod_factor, "ms")
            self.timer_play._interval = self.default_timer_speed/self.speed_mod_factor
            self.animation_freq_display_label.set_text(f"Taking action every {self.timer_play._interval*100//100/1000}s")



        if mode == "set":
            def handler(x):
                x = float(x)
                common(x)
            return handler
        elif mode == "+" or mode == "-":
            delta = self.speed_increment if mode == "+" else -self.speed_increment
            def handler():
                print(self.speed_mod_factor, delta, self.speed_mod_factor + delta)
                x = self.speed_mod_factor + delta
                if x > 0.0000000001 :
                    print("Changed Speed Modifier to",x)
                    common(x)
            return handler


    def run_sim(self) :
        self.timer_play.start()
    def stop_sim(self) :
        self.timer_play.stop()


    def get_inner_padded_rect(self, rect, pad ):
        padded_rect = []
        padded_rect.append( (rect[0][0] + pad, rect[0][1] + pad) )
        padded_rect.append( (rect[1][0] - pad, rect[1][1] + pad) )
        padded_rect.append( (rect[2][0] - pad, rect[2][1] - pad) )
        padded_rect.append( (rect[3][0] + pad, rect[3][1] - pad) )
        return padded_rect
        
    def draw_board(self, canvas):
        draw_start_time = time.time() # To compute FPS

        Vepsilonest = np.array(self.agent.estV)
        Vcolor = self.valuematrix2colormatrix(Vepsilonest)
        
        if self.show_Qval :
            Qest = np.array(self.agent.estQ)
            Qcolor = self.valuematrix2colormatrix(Qest)
            
        num_squares_along_height, num_squares_along_width = Vepsilonest.shape
        for i in range(num_squares_along_height):
            for j in range(num_squares_along_width):
                # Compute Cell dimensions
                rect = [
                    self.ij2xy(i, j),
                    self.ij2xy(i, j+1),
                    self.ij2xy(i+1, j+1),
                    self.ij2xy(i+1, j),
                ]
                             
                
                #Compute Qval Polygons
                if self.show_Qval :
                    inner_rect = self.get_inner_padded_rect(rect, self.cell_pad)
                    outer_rect = self.get_inner_padded_rect(rect, self.cell_pad/3)

                    left_action_polygon = [outer_rect[0], inner_rect[0], inner_rect[3], outer_rect[3]]
                    up_action_polygon = [outer_rect[0], inner_rect[0], inner_rect[1],outer_rect[1]]
                    right_action_polygon = [outer_rect[1], inner_rect[1], inner_rect[2],outer_rect[2]]
                    down_action_polygon = [outer_rect[2], inner_rect[2], inner_rect[3],outer_rect[3]]


                board_color = self.cmap.get(self.env.board[i, j], self.cmap["other"])
                canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        board_color
                    )
                
                if not (i,j) in self.board.done_tiles :
                    curr_color = f"rgb{tuple(Vcolor[i,j,:])}"
                    
                    #Outer trapezoids with respective QValues
                    if self.show_Qval :
                        # curr_action_colors = [ self.value2color(qval) for qval  in self.agent.estQ[i,j,:] ]

                        #Inner Square with the reward of the state
                        canvas.draw_polygon(
                            inner_rect, 0, 
                            self.grid_color, 
                            board_color
                        )
                            
                        canvas.draw_polygon(
                            left_action_polygon, self.grid_width/2,
                            self.grid_color,
                            f"rgb{tuple(Qcolor[ i,j,self.agent.action_to_index['left']])}"
                        )
                        canvas.draw_polygon(
                            right_action_polygon, self.grid_width/2,
                            self.grid_color,
                            f"rgb{tuple(Qcolor[ i,j,self.agent.action_to_index['right'] ])}"
                        )
                        canvas.draw_polygon(
                            up_action_polygon, self.grid_width/2,
                            self.grid_color,
                            f"rgb{tuple(Qcolor[ i,j,self.agent.action_to_index['up'] ])}"
                        )
                        canvas.draw_polygon(
                            down_action_polygon, self.grid_width/2,
                            self.grid_color,
                            f"rgb{tuple(Qcolor[ i,j,self.agent.action_to_index['down'] ])}"
                        )

                    #Borders of the cell
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        "rgba(0,0,0,0)"
                    )

                    #Inner Circle with Vvalue of the State
                    canvas.draw_circle(
                        self.ij2xy(i + 0.5, j + 0.5), 
                        self.cell_size*0.23,
                        self.grid_width/2, 
                        self.grid_color, 
                        curr_color
                    )


                    if self.show_Vepsilon_text :
                        canvas.draw_text(
                            "{:05.2f}".format(Vepsilonest[i,j]),
                            self.ij2xy(i+0.55, j+0.34),
                            font_size=self.text_size,
                            font_color= self.text_color
                        )
                    if self.draw_policy and board_color != self.wall_color :
                        opt_action = self.agent.index_to_action[self.agent.opt_policy[i][j]]
                        polygon = self.arrow_polygons[opt_action]
                        polygon = [self.ij2xy(i+di, j+dj) for di, dj in polygon]
                        canvas.draw_polygon(
                            polygon, 0, 
                            self.arrow_color, 
                            self.arrow_color
                            )
                else :
                    rect_done = [
                        self.ij2xy(i + 0.3, j+0.3),
                        self.ij2xy(i + 0.7, j+0.3),
                        self.ij2xy(i + 0.7, j+0.7),
                        self.ij2xy(i + 0.3, j+0.7),
                    ]
                    canvas.draw_polygon(
                        rect_done, 2.5, 
                        "white", 
                        "rgba(0, 0, 0, 0)"
                    )
                    
        i, j  = self.env.curr_pos
        canvas.draw_circle(
            self.ij2xy(i+0.5, j+0.5), 
            self.cell_size//4.5, 2, 
            f"rgb{self.agent_color}", f"rgb{self.agent_color}"
        )

        draw_end_time = time.time()
        self.fps_label.set_text(f"FPS : { int(1 / (draw_end_time - draw_start_time)) }")

    def valuematrix2colormatrix(self, Valmatrix):
        #Color Computation. Logic : Background Vector + scaled vector of linear interpolation between positive and negative. The scale is a cosine scale.
        #Resultant : Parametric curve going from approx positive to approx background to approx negative

        cmap = np.tanh(np.pi*Valmatrix/2)/2 + 0.5 
        cmap = np.expand_dims(cmap, -1)
        scaledcmap = 0.47*np.cos(2*np.pi*cmap) + 0.53
        cmap_linintrpol = self.cmap_bag_to_neg + cmap*(self.cmap_bag_to_pos - self.cmap_bag_to_neg)
        col_matrix = (self.background_color + cmap_linintrpol * scaledcmap).astype(int)
        return col_matrix
    
    def set_pad_l(self):
        num_squares_along_width, num_squares_along_height = self.env.board_width, self.env.board_height
        cell_w, cell_h = self.canvas_width//num_squares_along_width, self.canvas_height//num_squares_along_height
        cell_size = min(cell_w, cell_h)
        if cell_w>cell_h:
            x_pad = (self.canvas_width - num_squares_along_width*cell_size)//2
            y_pad = 0
        else:
            x_pad = 0
            y_pad = (self.canvas_height - num_squares_along_height*cell_size)//2
        
        self.x_pad = x_pad
        self.y_pad = y_pad
        self.cell_size = cell_size
        self.cell_pad = self.cell_pad_ratio*self.cell_size
        self.text_size = int(self.cell_size*0.15)

    def ij2xy(self, index0, index1):
        x = self.x_pad + index1*self.cell_size 
        y = self.y_pad + index0*self.cell_size
        return x, y

    def xy2ij(self, x, y, round=True):
        index1 = (x- self.x_pad)/self.cell_size
        index0 = (y- self.y_pad)/self.cell_size
        if round:
            index0 = math.floor(index0)
            index1 = math.floor(index1)
        return index0, index1
    

    def reset(self) :
        self.agent.reset_agent()
        self.env.reset()
    
    def single_step(self) :
        if self.single_step_running : #prevent concurrency issues where timer calls are faster than the runtime of the function
            return
        self.single_step_running = True
        
        if self.just_done : # Skips one frame so that we can see the agent hit the done state
            self.just_done = False
            self.env.reset()
            self.single_step_running = False
            return

        legal_actions, preferred_actions = self.env.get_legal_actions()
        if len(legal_actions) == 0 :
            raise("No actions available, the agent is blocked from all sides. Reconfigure the MDP environment")
            return
        
        if len(preferred_actions) != 0 and self.avoid_visited_states :
            available_actions = preferred_actions
        else :
            available_actions = legal_actions
        
        action = self.agent.select_action(self.env.curr_pos, available_actions)
        curr_state = self.env.curr_pos
        next_state, reward, done = self.env.step(action)
        self.agent.update_state(curr_state,action,next_state,reward, legal_actions)

        self.curr_T += 1
        if self.max_T_in_str != "" and self.curr_T == int(self.max_T_in_str) :
            done = True

        if done :
            self.just_done = True

        self.single_step_running = False
            

if __name__ == "__main__" :
    frame = simplegui.create_frame("Qlearning", 700,600) # There was one more argument, not sure what that is
    inputgui = MDPGUI(frame)
    qlearning_gui = Qlearning_with_GUI(frame)
    inputgui.set_control_transfer(qlearning_gui.take_over)
    qlearning_gui.set_control_transfer(inputgui.take_over)
    frame.start()