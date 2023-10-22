import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI,Board
import math
import time


def sigmoid(z):
    return 1/(1 + np.exp(-z))

class Agent :
    def __init__(self, state_space_shape, gamma = 0.9,  alpha = 0.1) :
        self.alpha = alpha #learning rate
        self.gamma = gamma
        self.state_space_shape = state_space_shape
        print(state_space_shape)
        self.action_to_index = {'left' : 0, 'right' : 1, 'up' : 2, 'down' :3}
        self.reset_agent()

    def reset_agent(self) :
        # self.estQ = np.random.random(size = self.state_space_shape + (4,) ) - 0.5 # The last indice stores action
        self.estQ = np.zeros(shape = self.state_space_shape + (4,)  )

    @property
    def board(self) :
        return self.estQ
    
    def select_action( self, state, available_actions, epsilon = 1) :
        toss = random.uniform(0,1)
        qvalues = []
        for action in available_actions :
            action_idx = self.action_to_index[action]
            qvalues.append(self.estQ[state[0], state[1], action_idx])

        if toss < epsilon :
            # We would like to explore more with higher value of epsilon
            return np.random.choice(available_actions)
        else :
            return available_actions[np.argmax(qvalues)]
    

    def update_state(self, state, action, new_state, reward) :
        # Q = (1 - alpha)*Q(st,at) + alpha(rt + gamma * max_a Q(st+1, a))
        action = self.action_to_index[action]
        Qlhs = (1 - self.alpha)* self.estQ[ state[0], state[1],action]
        Qrhs = self.alpha * ( reward + self.gamma * np.max(self.estQ[ new_state[0], new_state[1], : ]))
        self.estQ[state[0], state[1],action] = Qlhs + Qrhs
    

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
        
        if len(preferred_actions) == 0 :
            return legal_actions
        else :
            return preferred_actions

class Qlearning_with_GUI() :
    def __init__(self, frame, epsilon = 1) :
        self.frame = frame
        self.epsilon = epsilon
        
    def take_over(self, board, start_pos, transition_prob = 1) :
        board = Board(board)
        self.env = Mdp_env(board,start_pos=start_pos)
        self.agent = Agent((self.env.board_height, self.env.board_width))
        self.start_pos = start_pos
        self.setup_frame()
        self.set_pad_l()
        
    def setup_frame(self) :
        # Board visualisation constants
        
        self.grid_color = "blue"
        self.grid_width = 2
        self.draw_mode = 0
        self.cmap_negvval = np.array([224, 36, 36])
        self.cmap_posvval = np.array([39, 166, 39])
        self.cmap_neg_to_pos = self.cmap_posvval - self.cmap_negvval

        self.cmap = {
            0: "black",
            1: f"rgb{tuple(self.cmap_posvval)}",
            -1: f"rgb{tuple(self.cmap_negvval)}",
            -10: "grey",
            "other": "cyan"
        }
        print(self.cmap)
        # UI parameter imports
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height
        # Adding GUI elements
        frame.add_button("reset", self.reset)
        frame.add_button("Run Sim", self.run_sim)
        frame.add_button("Stop Sim", self.stop_sim)
        frame.add_button("Reset", self.reset)
        frame.set_draw_handler(self.draw_board)
        self.timer_play = simplegui.create_timer(1, self.single_step)


    def run_sim(self) :
        self.timer_play.start()
    def stop_sim(self) :
        self.timer_play.stop()

    def draw_board(self, canvas):
        Qest = self.agent.estQ
        Vest = (1 - self.epsilon)* np.max(Qest, axis=-1) + np.sum( (self.epsilon/4)*Qest,axis = -1 )
        cmap = np.tanh(np.pi*Vest/2)/2 + 0.5 #Parametric curve going from red to black to green, instead of just plain average
        num_squares_along_height, num_squares_along_width = Vest.shape
        for i in range(num_squares_along_height):
            for j in range(num_squares_along_width):
                rect = [
                    self.ij2xy(i, j),
                    self.ij2xy(i, j+1),
                    self.ij2xy(i+1, j+1),
                    self.ij2xy(i+1, j),
                ]
                color = self.cmap.get(self.env.board[i, j], self.cmap["other"])
                if color == "black" :
                    t = cmap[i,j]
                    curr_color = f"rgb{tuple( ( (t*self.cmap_posvval + (1-t)*self.cmap_negvval) *( 0.47*np.cos(2*np.pi*t) + 0.53) ).astype(int) )}"
                    print(curr_color)
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        curr_color
                    )
                    canvas.draw_text(
                        "%.2f"% Vest[i,j],
                        self.ij2xy(i+0.5, j+0.5),
                        font_size=12,
                        font_color="white"
                    )
                elif color == self.cmap["other"]:
                    canvas.draw_text(
                        str(int(self.env.board[i, j])),
                        self.ij2xy(i+0.5, j+0.5),
                        font_size=20,
                        font_color="black"
                    )
                else :
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        color
                    )
        i, j  = self.env.curr_pos
        canvas.draw_circle(
            self.ij2xy(i+0.5, j+0.5), 
            self.cell_size//4, 2, 
            "yellow", "yellow"
        )

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
        available_actions = self.env.get_legal_actions()
        if len(available_actions) == 0 :
            raise("No actions available")
            return
        
        action = self.agent.select_action(self.env.curr_pos, available_actions, self.epsilon)
        curr_state = self.env.curr_pos
        next_state, reward, done = self.env.step(action)
        self.agent.update_state(curr_state,action,next_state,reward)

        if done :
            # time.sleep(1)
            self.env.reset()

if __name__ == "__main__" :
    frame = simplegui.create_frame("Qlearning", 700,500) # There was one more argument, not sure what that is
    qlearning_gui = Qlearning_with_GUI(frame)
    inputgui = MDPGUI(frame, send_board_data_to = qlearning_gui.take_over)
    inputgui.start()