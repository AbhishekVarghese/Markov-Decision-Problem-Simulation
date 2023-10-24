import random, time
import numpy as np
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui 
import math
import json
import os
import copy


class DefaultDict(dict):
    def __init__(self, default_factory, **kwargs):
        super().__init__(**kwargs)

        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return self.default_factory()

class Board:
    def __init__(self, board, done_tiles):
        self.board = board
        self.height = board.shape[0]
        self.width = board.shape[1]
        self.shape = board.shape
        self.reward_dict = DefaultDict(lambda : 0)
        self.done_tiles = done_tiles
        self.blocked_tiles = []
        for i in range(self.height) :
            for j in range(self.width) :
                if not board[i][j] in (0,-10):
                    self.reward_dict[(i,j)] = board[i][j]
                elif board[i][j] == -10 :
                    self.blocked_tiles.append((i,j))


class MDPGUI:
    def __init__(self, frame):
        # first index is along width and second is along height
        self.frame = frame
        self.board = np.zeros((4, 4))
        self.done_tiles = []
        self.player_pos = (2, 0)
        m, n = self.board.shape
        self.transition_prob = 1.0

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

        self.current_board_name = "My Board"
        self.current_board_num = 0
        self.saved_boards = {
            self.current_board_name: {
                "board": self.board,
                "player_pos": self.player_pos,
                "done_tiles": self.done_tiles
            },
            "maze 1": 
            {
                "board": np.array(
                    [
                        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, 0, 0, 0, 0, -10, -10, -10, 0, 0, -10, -10, 0, -10, -10, 0, 0, 0, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, -10, 0, -10, -10, 0, 0, 0, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, 0, 0, 0, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, 0, 0, -10, 0, -10, 0, -10, -10, -10, 0, -10, 0, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, 0, 0, 0, -10, -10, 0, -10, 0, -10, -10, -10, 0, 0, 0, -10, 0, -10, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, 0, 0, -10, -10, -10, -10, 0, -10, 0, 0, -10, -10, 0, -10, -10, -10, 0, -10, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, 0, -10, -10, -10, 0, 0, -10, -10, 0, -10, -10, 0, -10, 0, 0, 0, -10, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, 0, -10, -10, 0, 0, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, -10, 0, 0, -10, 0, 0, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, 0, -10, -10, 0, -10, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, 0, 0, -10, -10, -10, 0, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, 0, 0, -10, -10, -10, -10, -10, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, 0, 0, 0, -10, 0, 0, -10, 0, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, -10, 0, -10, -10, -10, -10, -10, 0, -10, -10, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, 0, -10, -10, 0, -10, 0, 0, 0, -10, -10, 0, -10, 0, -10, -10, -10, -10, 0, 0, -10, 0, 0, 0, 0, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, 0, -10, 0, 0, -10, -10, -10, -10, -10, -10, 0, -10, 0, 0, -10, -10, -10, -10, 0, -10, 0, -10, 0, -10, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, 0, -10, 0, -10, -10, -10, 0, -10, -10, -10, 0, -10, -10, 0, 0, 0, -10, -10, 0, 0, 0, -10, 0, -10, -10, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, -10, -10, 0, -10, 0, -10, -10, 0, 0, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, -10, 0, 0, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, 0, 0, -10, 0, -10, 0, -10, -10, 0, -10, -10, -10, -10, 0, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, -10, 0, -10, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, -10, 0, -10, 0, -10, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, -10, 0, -10, -10, 0, -10, -10, -10, 0, -10, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, -10, 0, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, -10, -10, -10, -10, 0, -10, 0, 0, 0, 0, 0, -10, -10, 0, -10, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, -10, 0, -10, -10, 0, -10, -10, 0, -10, -10, 0, 0, -10, 0, 0, 0, 0, -10, -10, 0, 0, -10, -10, -10, -10, 0, -10, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, -10, -10, 0, -10, -10, 0, -10, -10, 0, -10, -10, 0, -10, -10, -10, -10, -10, 0, -10, -10, -10, 0, 0, 0, -10, 0, -10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, -10, -10, -10, -10, -10, 0, -10, -10, 0, -10, -10, 0, -10, -10, 0, -10, -10, -10, -10, -10, 0, -10, -10, 0, 0, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                    ]
                ),
                "player_pos": (10, 0),
                "done_tiles": []
            }
        }
        self.save_board_path = "./saved_boards.json"
        if os.path.exists(self.save_board_path):
            with open(self.save_board_path, "r") as f:
                other_saved_boards = json.load(f)
            for name in other_saved_boards.keys():
                other_saved_boards[name]["board"] = np.array(other_saved_boards[name]["board"])
                if name not in self.saved_boards:
                    self.saved_boards[name] = other_saved_boards[name]
                elif name == self.current_board_name:
                    self.saved_boards[name + " (1)"] = other_saved_boards[name]

        
        
        # UI parameter imports
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height
        
        self.setup_frame()

        
        # # # print(self.frame.__attr__)
        # all_fns = dir(self.frame)
        # fns = [f for f in all_fns if "set" in f]
        # print(fns)
        # # print(self.frame.add_input.__doc__)
        # print(dir(self.w_label))
        # print(dir(self.frame))

        self.x_pad, self.y_pad, self.l = self.get_pad_l()

    def setup_frame(self):
        m, n = self.board.shape
        # self.frame.add_label("Set Input Configurations")
        # self.frame.add_label("(Note: You need to press enter after every text input)")
        
        # self.frame.add_label("\n"*10)

        self.w_label = self.frame.add_input("Grid Width (Type or adjust)", self.board_input_handler(1), width=100)
        self.w_label.set_text(str(m))
        self.frame.add_button("+", self.board_button_handler(ax=1, delta=1), width = 100)
        self.frame.add_button("--", self.board_button_handler(ax=1, delta=-1), width = 100)
        self.h_label = self.frame.add_input("Grid Height (Type or adjust)", self.board_input_handler(0), width=100)
        self.h_label.set_text(str(n))
        self.frame.add_button("+", self.board_button_handler(ax=0, delta=1), width = 100)
        self.frame.add_button("--", self.board_button_handler(ax=0, delta=-1), width = 100)
        self.frame.set_draw_handler(self.draw_board)


        self.frame.add_label("\n"*10)

        self.frame.add_button("Draw +1 Reward", self.draw_mode_handler(1), width = 200)
        self.frame.add_button("Draw -1 Reward", self.draw_mode_handler(-1), width = 200)
        self.frame.add_button("Draw Wall", self.draw_mode_handler(-10), width = 200)
        # self.c_reward = self.frame.add_input("Draw Custom Reward", self.custom_draw_mode, width = 200)
        self.frame.add_button("Mark Done States", self.draw_mode_handler("done"), width = 200)
        self.frame.add_button("Erase", self.draw_mode_handler(0), width = 200)

        self.frame.add_label("\n"*10)
        self.frame.add_button("Set Start Position", self.set_start_pos, width = 200)


        # self.frame.add_label("\n"*10)
        # self.prob = self.frame.add_input("Probability of action execution", self.set_prob, width=100)
        # self.prob.set_text(str(self.transition_prob))

        self.frame.add_label("\n"*10)
        self.load_board_format = "Load Board: {}"
        self.load_board_label = self.frame.add_label(self.load_board_format.format(self.current_board_name), width = 200)
        self.frame.add_button("Next", self.load_board_handler(1), width = 100)
        self.frame.add_button("Prev", self.load_board_handler(-1), width = 100)

        self.frame.add_label("\n"*10)
        self.save_board_name = self.frame.add_input("Board Name", self.set_board_name, width=100)
        self.frame.add_button("Save Board", self.save_board, width = 200)
        # self.frame.add_button("Reset Saved Boards", self.reset_saved_boards, width = 200)

        self.frame.add_label("\n"*10)
        self.frame.add_button("Start Value Iteration", self.release_control("value_iteration"), width = 200)
        self.frame.add_button("Start Q Learning", self.release_control("q_learning"), width = 200)

        self.frame.set_mouseclick_handler(self.mouse_handler)
        self.frame.set_mousedrag_handler(self.mouse_handler)

    def take_over(self, board, player_pos):
        self.setup_frame()

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
            self.draw_mode = None
            self.frame._controls = []
            self.frame._draw_controlpanel()
            send_fn(
                Board(self.board,self.done_tiles), self.player_pos, 
                # self.transition_prob
            )
        return handler


    def start(self):
        try:
            self.frame.start()
        except AssertionError as e:
            print("Assertion error: ", e)
            for control in self.frame._controls:
                if hasattr(control, "_input_handler"):
                    control._input_pos = 0
            self.start()

    def stop(self,*args) :
        self.frame.stop()
    
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
        elif self.draw_mode is None:
            return
        elif self.draw_mode == "start_pos":
            self.player_pos = (i, j)
        elif self.draw_mode == "done":
            self.done_tiles.append((i, j))
        else:
            if self.draw_mode == 0 and (i, j) in self.done_tiles:
                self.done_tiles.remove((i, j))
            self.board[i, j] = self.draw_mode

    def board_button_handler(self, ax, delta):
        def handler():
            m, n = self.board.shape
            if ax == 0:
                m+=delta
                self.h_label.set_text(str(m))
            else:
                n+=delta
                self.w_label.set_text(str(n))
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

    def save_curr_board(self):
        self.saved_boards[self.current_board_name] = {
            "board": self.board,
            "player_pos": self.player_pos,
            "done_tiles": self.done_tiles,
        }


    def load_board_handler(self, delta):
        def handler():
            self.save_curr_board()
            self.current_board_num = (self.current_board_num + delta) % len(list(self.saved_boards.keys()))
            self.current_board_name = list(self.saved_boards.keys())[self.current_board_num]
            board_config = self.saved_boards[self.current_board_name]
            self.board, self.player_pos, self.done_tiles = board_config["board"], board_config["player_pos"], board_config["done_tiles"]
            self.done_tiles = [tuple(t) for t in self.done_tiles]
            self.x_pad, self.y_pad, self.l = self.get_pad_l()
            self.load_board_label.set_text(self.load_board_format.format(self.current_board_name)) 
            m, n = self.board.shape
            self.w_label.set_text(str(m))
            self.h_label.set_text(str(n))
        return handler

    def set_board_name(self, x):
        x = str(x)
        self.saved_boards = {
            name if name!= self.current_board_name else x:self.saved_boards[name] 
            for name in self.saved_boards.keys()
        }
        self.current_board_name = x
        self.load_board_label.set_text(self.load_board_format.format(self.current_board_name)) 
        
    def save_board(self):
        self.save_curr_board()
        saved_boards_list = copy.deepcopy(self.saved_boards)
        for b in saved_boards_list:
            saved_boards_list[b]["board"] = saved_boards_list[b]["board"].tolist()
        s = json.dumps(saved_boards_list, indent=4)
        s = s.replace(",\n                ", ", ")
        s = s.replace("[\n                ", "[")
        s = s.replace("\n            ]", "]")
        with open(self.save_board_path, "w") as f:
            f.write(s)

    def reset_saved_boards(self):
        os.remove(self.save_board_path)


    def update_board(self, m, n):
        cur_m, cur_n = self.board.shape
        new_board = np.zeros((m, n))
        for i in range(min(cur_m, m)):
            for j in range(min(cur_n, n)):
                new_board[i, j] = self.board[i, j]
        i, j = self.player_pos
        def check_pos(l):
            i, j = l 
            return i < m and j < n and i >= 0 and j >= 0
        if not check_pos(self.player_pos):
            self.player_pos = (0, 0)
        self.board = new_board
        self.done_tiles = [pos for pos in self.done_tiles if check_pos(pos)]

        self.x_pad, self.y_pad, self.l = self.get_pad_l()

    def get_pad_l(self):
        m, n = self.board.shape
        w, h = self.canvas_width//n, self.canvas_height//m
        l = min(w, h)
        if w>h:
            x_pad = (self.canvas_width - n*l)//2
            y_pad = 0
        else:
            x_pad = 0
            y_pad = (self.canvas_height - m*l)//2
        return x_pad, y_pad, l

    def ij2xy(self, i, j):
        x = self.x_pad + j*self.l 
        y = self.y_pad + i*self.l
        return x, y

    def xy2ij(self, x, y, round=True):
        j = (x-self.x_pad)/self.l
        i = (y-self.y_pad)/self.l
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
                if (i, j) in self.done_tiles:
                    rect = [
                        self.ij2xy(i + 0.3, j+0.3),
                        self.ij2xy(i + 0.7, j+0.3),
                        self.ij2xy(i + 0.7, j+0.7),
                        self.ij2xy(i + 0.3, j+0.7),
                    ]
                    canvas.draw_polygon(
                        rect, 2, 
                        "white", 
                        "rgba(0, 0, 0, 0)"
                    )

        i, j,  = self.player_pos
        _, _, cell_size = self.get_pad_l()
        canvas.draw_circle(
            self.ij2xy(i+0.5, j+0.5), 
            cell_size//4, 2, 
            "yellow", "yellow"
        )


if __name__ == "__main__" :
    canvas_width = 700
    canvas_height = 600
    frame = simplegui.create_frame("MDP Visualization - Set Input Configurations", canvas_width, canvas_height)
    mdp = MDPGUI(frame)
    mdp.start()