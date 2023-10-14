import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui

# from input_mdp import MDPGUI,Board


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
        self.board_length = board.length
        self.board_width = board.width
        self.reward_dict = board.reward_dict
        self.reward_tiles = board.reward_dict.keys()
        self.blocked_tiles = board.blocked_tiles
        self.done_tiles = board.done_tiles
        if start_pos :
            self.curr_pos = start_pos
        else :
            all_tiles = set(itertools.product(range(self.board_length, self.board_width)))
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
            if self.curr_pos[0] < self.board_length and not new_pos in self.blocked_tiles :
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
        if self.curr_pos[0] < self.board_length and not (self.curr_pos[0] + 1, self.curr_pos[1]) in self.blocked_tiles :
            legal_actions.append('right')
        if self.curr_pos[0] >= 0 and not (self.curr_pos[0], self.curr_pos[1] - 1) in self.blocked_tiles :
            legal_actions.append('up')
        if self.curr_pos[0] < self.board_width and not (self.curr_pos[0], self.curr_pos[1] + 1) in self.blocked_tiles :
            legal_actions.append('down')
        
        return legal_actions


class Qlearning_with_GUI() :
    def __init__(self, board, start_pos, window_width = 700, window_height=500, epsilon = 0.25) :
        self.env = Mdp_env(board,start_pos=start_pos)
        self.agent = Agent((self.env.board_length, self.env.board_width))
        self.start_pos = start_pos
        self.epsilon = epsilon
        self.setup_frame( window_width, window_height)
        # Start the frame animation
        self.frame.start()

    def setup_frame(self,window_width, window_height) :

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

        frame = simplegui.create_frame("Qlearning", window_width, window_height) # There was one more argument, not sure what that is
        frame.add_button("reset", self.reset)
        frame.add_button("Run", self.run)
        frame.add_button("Stop", self.stop)
        frame.add_button("Reset", self.reset)
        frame.set_draw_handler(self.draw_board)

        self.frame = frame
        self.timer_play = simplegui.create_timer(500, self.single_step)


        
    def draw_board(self, canvas):
        m, n = self.agent.board.shape
        for i in range(m):
            for j in range(n):
                rect = [
                    self.ij2xy(i, j),
                    self.ij2xy(i, j+1),
                    self.ij2xy(i+1, j+1),
                    self.ij2xy(i+1, j),
                ]
                color = self.cmap.get(self.board[i, j], self.cmap["other"])
                canvas.draw_polygon(
                    rect, self.grid_width, 
                    self.grid_color, 
                    color
                )
                if color == self.cmap["other"]:
                    canvas.draw_text(
                        str(int(self.board[i, j])),
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
    inputgui = MDPGUI()
    board = Board(inputgui.board)
    del inputgui
    start_pos = (0,0)
    Qlearning_with_GUI(board,start_pos)