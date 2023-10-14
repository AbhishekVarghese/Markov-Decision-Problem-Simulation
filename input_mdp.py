import random, time
import numpy as np
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 
import math


class MDPGUI:
    def __init__(self):
        # first index is along width and second is along height
        self.board = np.zeros((8, 4))
        self.player_pos = (2, 0)
        m, n = self.board.shape
        self.transition_prob = 0.8

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
        
        # UI initialisations
        self.canvas_width = 700
        self.canvas_height = 500

        self.frame = simplegui.create_frame("MDP Visualization", self.canvas_width, self.canvas_height)

        self.frame.add_label("Set Input Configurations")
        self.frame.add_label("(Note: You need to press enter after every text input)")
        
        self.frame.add_label("\n"*10)

        self.w_label = self.frame.add_input("Grid Width (Type or adjust)", self.board_input_handler(0), width=100)
        self.w_label.set_text(str(m))
        self.frame.add_button("+", self.board_button_handler(ax=0, delta=1), width = 100)
        self.frame.add_button("--", self.board_button_handler(ax=0, delta=-1), width = 100)
        self.h_label = self.frame.add_input("Grid Height (Type or adjust)", self.board_input_handler(1), width=100)
        self.h_label.set_text(str(n))
        self.frame.add_button("+", self.board_button_handler(ax=1, delta=1), width = 100)
        self.frame.add_button("--", self.board_button_handler(ax=1, delta=-1), width = 100)
        self.frame.set_draw_handler(self.draw_board)


        self.frame.add_label("\n"*10)

        self.frame.add_button("Draw +1 Reward", self.draw_mode_handler(1), width = 200)
        self.frame.add_button("Draw -1 Reward", self.draw_mode_handler(-1), width = 200)
        self.frame.add_button("Draw Wall", self.draw_mode_handler(-10), width = 200)
        self.c_reward = self.frame.add_input("Draw Custom Reward", self.custom_draw_mode, width = 200)
        self.frame.add_button("Erase Reward", self.draw_mode_handler(0), width = 200)

        self.frame.add_label("\n"*10)
        self.frame.add_button("Set Start Position", self.set_start_pos, width = 200)


        self.frame.add_label("\n"*10)
        self.prob = self.frame.add_input("Probability of action execution", self.set_prob, width=100)
        self.prob.set_text(str(self.transition_prob))

        self.frame.set_mouseclick_handler(self.mouse_handler)
        self.frame.set_mousedrag_handler(self.mouse_handler)
        # # # print(self.frame.__attr__)
        # all_fns = dir(self.frame)
        # fns = [f for f in all_fns if "set" in f]
        # print(fns)
        # # print(self.frame.add_input.__doc__)

    def start(self):
        try:
            self.frame.start()
        except AssertionError as e:
            print("Assertion error: ", e)
            self.w_label._input_pos = 0
            self.h_label._input_pos = 0
            self.c_reward._input_pos = 0
            self.prob._input_pos = 0
            self.start()

    def set_start_pos(self):
        self.draw_mode = "start_pos"

    def set_prob(self, p):
        self.transition_prob = p

    def draw_mode_handler(self, mode):
        def handler():
            self.draw_mode = mode
        return handler

    def custom_draw_mode(self, mode):
        self.draw_mode = int(mode)
    
    def mouse_handler(self, pos):
        x, y = pos
        i, j = self.xy2ij(x, y)
        if i >= self.board.shape[0] or j >= self.board.shape[1] or i < 0 or j < 0:
            return
        if self.draw_mode == "start_pos":
            self.player_pos = (i, j)
        else:
            self.board[i, j] = self.draw_mode

    def board_button_handler(self, ax, delta):
        def handler():
            m, n = self.board.shape
            if ax == 0:
                m+=delta
                self.w_label.set_text(str(m))
            else:
                n+=delta
                self.h_label.set_text(str(n))
            assert m>0 and n>0, "Either width or height has become zero"
            self.update_board(m, n)
        return handler
    
    def board_input_handler(self, ax):
        def handler(i):
            m, n = self.board.shape
            i = int(i)
            if ax == 0:
                m = i
            else:
                n = i
            assert m>0 and n>0, "Either width or height has become zero"
            self.update_board(m, n)
        return handler


    def update_board(self, m, n):
        cur_m, cur_n = self.board.shape
        new_board = np.zeros((m, n))
        for i in range(min(cur_m, m)):
            for j in range(min(cur_n, n)):
                new_board[i, j] = self.board[i, j]
        i, j = self.player_pos
        if i >= m or j >= n or i < 0 or j < 0:
            self.player_pos = (0, 0)
        self.board = new_board

    def get_pad_l(self):
        m, n = self.board.shape
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


    def draw_board(self, canvas):
        m, n = self.board.shape
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

mdp = MDPGUI()
mdp.start()