import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI,Board
import math

class Agent :
    def __init__(self, state_space_shape, gamma = 0.9,  alpha = 0.01) :
        self.alpha = alpha #learning rate
        self.gamma = gamma
        self.state_space_shape = state_space_shape
        self.reset_agent()
        self.action_to_index = {'left' : 0, 'right' : 1, 'up' : 2, 'down' :3}
        self.reset_agent()

    def reset_agent(self) :
        self.estQ = np.random.random(size = self.state_space_shape + (4,) ) # The last indice stores action

    @property
    def board(self) :
        return self.estQ
    
    def select_action( self, state, available_actions, epsilon = 1) :
        toss = random.uniform(0,1)
        qvalues = []
        for action in available_actions :
            action_idx = self.action_to_index[action]
            qvalues.append(self.estQ[*state, action_idx])

        if toss < epsilon :
            # We would like to explore more with higher value of epsilon
            return np.random.choice(available_actions)
        else :
            return available_actions[np.argmax(qvalues)]
    

    def update_state(self, state, action, new_state, reward) :
        # Q = (1 - alpha)*Q(st,at) + alpha(rt + gamma * max_a Q(st+1, a))

        Qlhs = (1 - self.alpha)* self.state_space[ *state,action]
        Qrhs = self.alpha( reward + self.gamma * np.max(self.estQ[ *new_state, : ]))
        self.estQ[*state,action] = Qlhs + Qrhs
    

class Mdp_env :
    def __init__(self, board, start_pos = None) :
        self.board_height = board.height
        self.board_width = board.width
        self.reward_dict = board.reward_dict
        self.reward_tiles = board.reward_dict.keys()
        self.blocked_tiles = board.blocked_tiles
        self.done_tiles = board.done_tiles
        self.board = board.board
        if start_pos :
            self.curr_pos = start_pos
        else :
            all_tiles = set(itertools.product(range(self.board_height, self.board_width)))
            available_tiles = (all_tiles - set(self.reward_tiles)) - set(self.blocked_tiles)
            self.curr_pos = np.random.choice(available_tiles)
    

    def step(self, action) :
        # This piece of code, although repetative, would save memory and speed for huge board dimensions
        if action == 'left' :
            new_pos = (self.curr_pos[0] - 1, self.curr_pos[1])
            if self.curr_pos[0] >= 0 and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go left from this position {self.curr_pos}")
        elif action == 'right' :
            new_pos = (self.curr_pos[0] + 1, self.curr_pos[1])
            if self.curr_pos[0] < self.board_height and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go right from this position {self.curr_pos}")
        elif action == 'up' :
            new_pos = (self.curr_pos[0], self.curr_pos[1] - 1)
            if self.curr_pos[0] >= 0 and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go up from this position {self.curr_pos}")
        elif action == 'down' :
            new_pos = (self.curr_pos[0], self.curr_pos[1] + 1)
            if self.curr_pos[0] < self.board_width and not new_pos in self.blocked_tiles :
                self.curr_pos = new_pos
            else :
                warnings.warn(f"You can't go down from this position {self.curr_pos}")
        
        reward  = self.reward_dict[self.curr_pos]
        if self.curr_pos in self.done_tiles :
            done = True

        return self.curr_pos, reward, done

    def get_legal_actions(self) :
        legal_actions = []
        if self.curr_pos[0] >= 0 and not (self.curr_pos[0] - 1, self.curr_pos[1]) in self.blocked_tiles :
            legal_actions.append('left')
        if self.curr_pos[0] < self.board_height and not (self.curr_pos[0] + 1, self.curr_pos[1]) in self.blocked_tiles :
            legal_actions.append('right')
        if self.curr_pos[0] >= 0 and not (self.curr_pos[0], self.curr_pos[1] - 1) in self.blocked_tiles :
            legal_actions.append('up')
        if self.curr_pos[0] < self.board_width and not (self.curr_pos[0], self.curr_pos[1] + 1) in self.blocked_tiles :
            legal_actions.append('down')
        
        return legal_actions


class Qlearning_with_GUI() :
    def __init__(self, frame, epsilon = 0.25) :
        self.frame = frame
        self.epsilon = epsilon
        
    def take_over(self, board, start_pos) :
        board = Board(board)
        self.env = Mdp_env(board,start_pos=start_pos)
        self.agent = Agent((self.env.board_height, self.env.board_width))
        self.start_pos = start_pos
        self.setup_frame()
        
    def setup_frame(self) :
        # Board visualisation constants
        self.cmap = {
            0: "black",
            1: "green",
            -1: "red",
            -10: "grey",
            "other": "cyan"
        }
        self.grid_color = "blue"
        self.grid_width = 2
        self.draw_mode = 0

        # UI parameter imports
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height
        # Adding GUI elements
        frame.add_button("reset", self.reset)
        frame.add_button("Run Sim", self.run_sim)
        frame.add_button("Stop Sim", self.stop_sim)
        frame.add_button("Reset", self.reset)
        frame.set_draw_handler(self.draw_board)
        self.timer_play = simplegui.create_timer(1000, self.single_step)


    def run_sim(self) :
        self.timer_play.start()
    def stop_sim(self) :
        self.timer_play.stop()

    def draw_board(self, canvas):
        Qest = self.agent.estQ
        Vest = (1 - self.epsilon)* np.max(Qest, axis=-1) + np.sum( (self.epsilon/4)*Qest,axis = -1 )
        m, n = Vest.shape
        for i in range(m):
            for j in range(n):
                rect = [
                    self.ij2xy(i, j),
                    self.ij2xy(i, j+1),
                    self.ij2xy(i+1, j+1),
                    self.ij2xy(i+1, j),
                ]
                color = self.cmap.get(self.env.board[i, j], self.cmap["other"])
                canvas.draw_polygon(
                    rect, self.grid_width, 
                    self.grid_color, 
                    color
                )
                if color == self.cmap["other"]:
                    canvas.draw_text(
                        str(int(self.env.board[i, j])),
                        self.ij2xy(i+0.5, j+0.5),
                        font_size=20,
                        font_color="black"
                    )
        i, j,  = self.player_pos
        _, _, cell_size = self.get_pad_l()
        canvas.draw_circle(
            self.ij2xy(i+0.5, j+0.5), 
            cell_size//4, 2, 
            "yellow", "yellow"
        )

    def get_pad_l(self):
        m, n = self.env.board_width, self.env.board_height
        w, h = self.canvas_width//m, self.canvas_height//n
        l = min(w, h)
        if w>h:
            x_pad = (self.canvas_width - m*l)//2
            y_pad = 0
        else:
            x_pad = 0
            y_pad = (self.canvas_height - n*l)//2
        return x_pad, y_pad, l

    def ij2xy(self, i, j):
        x_pad, y_pad, l = self.get_pad_l()

        x = x_pad + i*l 
        y = y_pad + j*l
        return x, y

    def xy2ij(self, x, y, round=True):
        x_pad, y_pad, l = self.get_pad_l()

        i = (x-x_pad)/l
        j = (y-y_pad)/l
        if round:
            i = math.floor(i)
            j = math.floor(j)
        return i, j
    

    def reset(self) :
        self.agent.reset_agent()
        self.env.reset()
    
    def single_step(self) :
        available_actions = self.env.get_legal_actions()
        if len(available_actions) == 0 :
            return
        
        action = self.agent.select_action(self.env.curr_pos, available_actions, self.epsilon)
        curr_state = self.env.curr_pos
        next_state, reward, done = self.env.step(action)
        self.agent.update_state(curr_state,action,next_state,reward)

        if done :
            self.env.reset()

if __name__ == "__main__" :
    frame = simplegui.create_frame("Qlearning", 700,500) # There was one more argument, not sure what that is
    qlearning_gui = Qlearning_with_GUI(frame)
    inputgui = MDPGUI(frame, send_board_data_to = qlearning_gui.take_over)
    frame.start()