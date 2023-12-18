import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI,Board
import math
import time

class ValueIteration():
    def __init__(self, rewards, done_tiles, wall_reward=-10):
        self.rewards = rewards
        self.done_tiles = done_tiles
        
        self.wall_reward = wall_reward
        self.iteration2values = []
        self.current_iter = 0

    def shift_arr(self, arr, direction, min_possible):
        if direction == "right":
            shifted = np.pad(
                arr[:, 1:], 
                pad_width=((0, 0), (0, 1)),
                constant_values=min_possible
            )
        elif direction == "left":
            shifted = np.pad(
                arr[:, :-1], 
                pad_width=((0, 0), (1, 0)),
                constant_values=min_possible
            )
        elif direction == "up":
            shifted = np.pad(
                arr[1:, :], 
                pad_width=((0, 1), (0, 0)),
                constant_values=min_possible
            )
        elif direction == "down":
            shifted = np.pad(
                arr[:-1, :], 
                pad_width=((1, 0), (0, 0)),
                constant_values=min_possible
            )
        return shifted

    def reset(self):
        self.iteration2values = []

    def run(self, max_iters, discount, p, reset=False):
        self.max_iters = max_iters
        self.current_iter = 0
        min_possible = np.min(self.rewards)

        # values = self.rewards.copy()
        # values[values == self.wall_reward] = min_possible
        # values[values == 0] = min_possible
        # values = min_possible*np.ones(self.rewards.shape)
        values = np.zeros(self.rewards.shape)


        if reset:
            self.reset()

        wall_mask = self.rewards == self.wall_reward

        directions = ["left", "right", "up", "down"]
        rewards_shifted = {
            di: self.shift_arr(self.rewards, di, self.wall_reward) for di in directions
        }

        if len(self.iteration2values) == 0:
            self.iteration2values.append(values)
        for i in range(1, max_iters+1):
            if i < len(self.iteration2values):
                continue
            self.current_iter = i 

            prev_values = self.iteration2values[i-1].copy()

            prev_shifted = {
                di:self.shift_arr(prev_values, di, 0) for di in directions
            }

            proposals = {
                di: rewards_shifted[di] + discount * prev_shifted[di]
                for di in directions
            }
            for di in proposals.keys():
                for tile in self.done_tiles:
                    i, j = tile
                    proposals[di][i, j] = self.rewards[i, j] + discount * prev_values[i, j]

            new_values = [
                np.sum(np.stack([
                    p * proposals[di2] if di2 == di else (1-p)/3 * proposals[di2]
                    for di2 in directions
                ], axis=0), axis=0)
                for di in directions
            ]
            # Action Stay
            # new_values.append(
            #     self.rewards + discount * prev_values
            # )

            curr_values = np.stack(
                new_values, axis=0
            )
            curr_values = curr_values.max(axis=0)
            curr_values += self.rewards
            curr_values[wall_mask] = self.wall_reward
            self.iteration2values.append(
                curr_values
            )


class ValueIterationGUI(MDPGUI):
    def __init__(self, frame, discount=0.9) :
        self.frame = frame
        self.discount = discount
        self.discount_increment = 0.05

        self.iteration = 20
        self.iteration_increment = 1
        self.maxiter_increment = 20

        self.timer_speed = 1
        self.speed_increment = 1
        self.timer_multiplier = 50

        self.grid_color = "black"
        self.text_color = "black"
        self.wall_color = "grey"

        self.cmap_negvval = np.array([199, 21, 133])
        self.cmap_posvval = np.array([0, 250, 154])
        self.background_color = np.array([248, 248, 255]) 
        self.cmap = {
            0: f"rgb{tuple(self.background_color)}",
            1: f"rgb{tuple(self.cmap_posvval)}",
            -1: f"rgb{tuple(self.cmap_negvval)}",
            -10: self.wall_color,
            "other": "cyan"
        }
        self.text_size = 12
        self.grid_width = 2
        self.draw_mode = 0
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height

        self.draw_policy = False
        self.draw_agent_path = False
        self.draw_values = False
        self.agent_path = []

        self.arrow_color = "black"
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
            ]
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

    def take_over(self, board, player_pos, p=1):
        self.p = p
        self.rewards = board.board
        self.raw_done_tiles = board.done_tiles
        self.done_tiles = []
        m, n = self.rewards.shape
        for i in range(m):
            for j in range(n):
                if self.rewards[i, j] != 0 and self.rewards[i, j] != -10:
                    self.done_tiles.append((i, j))

        self.board = board.board
        self.player_pos = player_pos
        self.x_pad, self.y_pad, self.l = self.get_pad_l()
        self.text_size = int(self.l*0.2)

        self.d_label = self.frame.add_input("Discount Factor (Gamma)", self.update_discount("set"), width=100)
        self.d_label.set_text(str(self.discount))
        self.frame.add_button("+", self.update_discount("+"), width = 100)
        self.frame.add_button("--", self.update_discount("-"), width = 100)
        
        self.frame.add_label("\n"*10)

        self.it_label = self.frame.add_input("Iteration", self.update_iteration("set"), width=100)
        self.it_label.set_text(str(self.iteration))
        self.frame.add_button("+", self.update_iteration("+"), width = 100)
        self.frame.add_button("--", self.update_iteration("-"), width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Show all iterations", self.showall_iteration, width = 100)
        self.speed_label = self.frame.add_input("Speed", self.update_speed("set"), width=100)
        self.speed_label.set_text(str(self.timer_speed))
        self.frame.add_button("+", self.update_speed("+"), width = 100)
        self.frame.add_button("--", self.update_speed("-"), width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Show Policy", self.show_policy, width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Show Agent Path", self.show_agent_path, width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Show Values", self.show_values, width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Back to input", self.release_control("input"), width = 200)
        self.frame.add_button("Switch to Q - learning", self.release_control("q_learning"), width = 200)

        self.algorithm = ValueIteration(
            self.rewards, self.done_tiles
        )
        self.algorithm.run(self.iteration, self.discount, self.p)
        self.update_values()

        self.draw_status = None
        self.frame.set_draw_handler(self.draw_board)

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
            self.stop_timer()
            self.draw_mode = None
            self.frame._controls = []
            self.frame._draw_controlpanel()
            send_fn(
                Board(self.board,self.raw_done_tiles), self.player_pos, 
            )
        return handler

    def update_values(self, iteration=None):
        if iteration is None:
            iteration = self.iteration
        self.values = self.algorithm.iteration2values[iteration].copy()
        mx, mn = self.values.max(), self.values.min()

        # # adjust minimum (only for better scaling)
        # if len(np.unique(self.values)) > 2:
        #     new_val = self.values.copy()
        #     new_val[self.values == mn] = mx
        #     second_mn = new_val.min()
        #     self.values[self.values == mn] = second_mn
        #     mx, mn = self.values.max(), self.values.min()

        # self.values = (self.values - mn) / (mx - mn) if mx != mn else np.ones(self.values.shape)
        self.update_policy()

    def update_policy(self):
        m, n = self.board.shape
        self.policy = [[0 for _ in range(n)] for _ in range(m)]
        def val_sf(i_, j_, default=self.values.min()-1):
            if (i_ < m and i_ >= 0 and j_ < n and j_ >= 0):
                val = self.values[i_, j_] 
            else:
                val = default
            return val

        for i in range(m):
            for j in range(n):
                stay = self.values[i, j]
                r, l, u, d = val_sf(i, j+1), val_sf(i, j-1), val_sf(i-1, j), val_sf(i+1, j)
                max_val = max(stay, l, r, u, d)
                if (max_val == stay and max_val not in [l, r, u, d]) or (i, j) in self.done_tiles:
                    self.policy[i][j] = "stay"
                elif max_val == l:
                    self.policy[i][j] = "left"
                elif max_val == r:
                    self.policy[i][j] = "right"
                elif max_val == u:
                    self.policy[i][j] = "up"
                elif max_val == d:
                    self.policy[i][j] = "down"

        self.agent_path = []
        self.draw_agent_path = False


    def update_discount(self, mode):
        def common(x):
            self.stop_timer()
            assert (x >= 0) and (x <= 1), "Error: Discount going out of bounds"
            self.discount = x
            self.d_label.set_text("{:.3f}".format(self.discount))
            self.algorithm.run(self.iteration, self.discount, self.p, reset=True)
            draw_agent_path = self.draw_agent_path
            self.update_values()
            if draw_agent_path:
                self.show_agent_path()
                

        if mode == "set":
            def handler(x):
                x = float(x)    
                common(x)
            return handler
        elif mode == "+" or mode == "-":
            delta = self.discount_increment if mode == "+" else -self.discount_increment
            def handler():
                x = self.discount + delta
                common(x)
            return handler

    def update_iteration(self, mode):
        def common(x):
            self.stop_timer()
            assert (x >= 0), "Error: Iteration has to be >=0"
            self.iteration = x
            self.it_label.set_text(str(self.iteration))
            if self.iteration > self.algorithm.max_iters:
                new_max_iter = (self.iteration//self.maxiter_increment + 1) * self.maxiter_increment
                self.algorithm.run(new_max_iter, self.discount, self.p)
            draw_agent_path = self.draw_agent_path
            self.update_values()
            if draw_agent_path:
                self.show_agent_path()

        if mode == "set":
            def handler(x):
                x = int(float(x))
                common(x)
            return handler
        elif mode == "+" or mode == "-":
            delta = self.iteration_increment if mode == "+" else -self.iteration_increment
            def handler():
                x = self.iteration + delta
                common(x)
            return handler

    def update_speed(self, mode):
        def common(x):
            assert (x >= 0), "Error: Speed has to be >=0"
            self.timer_speed = x
            self.speed_label.set_text("{:.3f}".format(self.timer_speed))
            try:
                self.timer_play._interval = max(1, self.timer_multiplier // self.timer_speed)
            except AttributeError:
                return


        if mode == "set":
            def handler(x):
                x = float(x)
                common(x)
            return handler
        elif mode == "+" or mode == "-":
            delta = self.speed_increment if mode == "+" else -self.speed_increment
            def handler():
                x = self.timer_speed + delta
                common(x)
            return handler

    def stop_timer(self):
        if hasattr(self, "timer_play") and self.timer_play.is_running:
            self.timer_play.stop()
        self.draw_status = None


    def value_it_step(self):
        if self.intermediate_iter > self.iteration:
            self.timer_play.stop()
            self.draw_status = None
            return
        
        self.update_values(self.intermediate_iter)
        self.draw_status = "Iteration {:3d}/{:3d}".format(self.intermediate_iter, self.iteration)

        self.intermediate_iter += 1

    def showall_iteration(self):
        self.intermediate_iter = 1
        interval = max(1, self.timer_multiplier // self.timer_speed)
        self.timer_play = simplegui.create_timer(interval, self.value_it_step)
        self.timer_play.start()
        
    def show_policy(self):
        self.draw_policy = not self.draw_policy 

    def show_agent_path(self):
        self.draw_agent_path = not self.draw_agent_path
        if self.draw_agent_path:
            self.agent_path = []
            i, j = self.player_pos
            self.agent_path.append((i, j))
            transition = {
                "left": (0, -1),
                "right": (0, 1),
                "up": (-1, 0),
                "down": (1, 0),
            }
            m, n = self.board.shape
            steps = m*n+1
            while self.policy[i][j] != "stay" and steps > 0:
                di, dj = transition[self.policy[i][j]]
                i += di
                j += dj
                self.agent_path.append((i, j))
                steps -= 1

    def show_values(self):
        self.draw_values = not self.draw_values

    def value2color(self, val, max_val, min_val):
        if val < 0:
            color = abs(val)/abs(min_val+1e-10) * self.cmap_negvval
        else:
            color = abs(val)/abs(max_val+1e-10) * self.cmap_posvval
        color = color.tolist()
        color = [int(c) for c in color]
        color_str = "rgb({}, {}, {})".format(*color)
        return color_str


    def value2color_vectorized(self, val_matrix):
        val_matrix = val_matrix[:, :, None]
        max_val, min_val = self.values.max(), self.values.min()
        colors = (val_matrix <= 0) * (self.background_color[None, None, :] + (self.cmap_negvval[None, None, :] - self.background_color[None, None, :]) * np.abs(val_matrix/(min_val+1e-10))) \
                + (val_matrix > 0) * (self.background_color[None, None, :] + (self.cmap_posvval[None, None, :] - self.background_color[None, None, :]) * np.abs(val_matrix/(max_val+1e-10)))
        colors = np.uint8(colors)
        return colors

    def draw_board(self, canvas):
        m, n = self.board.shape
        color_matrix = self.value2color_vectorized(self.values)
        # max_val, min_val = self.values.max(), self.values.min()
        for i in range(m):
            for j in range(n):
                rect = [
                    self.ij2xy(i, j),
                    self.ij2xy(i, j+1),
                    self.ij2xy(i+1, j+1),
                    self.ij2xy(i+1, j),
                ]
                color = self.cmap.get(self.board[i, j], self.cmap["other"])
                if color != self.wall_color:
                    # value_color = "rgb({0}, {0}, {0})".format(int(255*self.values[i, j]))
                    value_color = f"rgb{tuple(color_matrix[i, j, :])}"
                    if self.draw_agent_path and (i, j) in self.agent_path:
                        value_color = "yellow"
                    
                    # canvas.draw_polygon(
                    #     rect, self.grid_width, 
                    #     self.grid_color, 
                    #     value_color
                    # )

                    background_color = "rgba(0,0,0,0)" if color == f"rgb{tuple(self.background_color)}" else color
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        value_color
                    )

                    if self.draw_values:
                        canvas.draw_text(
                                "{:05.2f}".format(self.values[i, j]),
                                self.ij2xy(i+0.55, j+0.27),
                                font_size=self.text_size,
                                font_color=self.text_color
                            )


                else :
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        color
                    )
                if self.draw_policy and color != self.wall_color:
                    if self.policy[i][j] in self.arrow_polygons.keys():
                        polygon = self.arrow_polygons[self.policy[i][j]]
                        polygon = [self.ij2xy(i+di, j+dj) for di, dj in polygon]
                        canvas.draw_polygon(
                            polygon, 0, 
                            self.arrow_color, 
                            self.arrow_color
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

        if not (self.draw_policy or self.draw_agent_path):
            i, j  = self.player_pos
            canvas.draw_circle(
                self.ij2xy(i+0.5, j+0.5), 
                self.l//4, 2, 
                "yellow", "yellow"
            )
            if self.draw_values:
                canvas.draw_text(
                    "{:05.2f}".format(self.values[i, j]),
                    self.ij2xy(i+0.55, j+0.27),
                    font_size=self.text_size,
                    font_color=self.text_color
                )
        if self.draw_status is not None:
            self.frame._canvas.draw_text(
                self.draw_status,
                (20, 30),
                font_size=20,
                font_color=self.text_color
            )


if __name__ == "__main__" :
    frame = simplegui.create_frame("Value Iteration", 700,600)
    inputgui = MDPGUI(frame)
    value_it_gui = ValueIterationGUI(frame)
    inputgui.set_control_transfer(value_it_gui.take_over)
    value_it_gui.set_control_transfer(inputgui.take_over)
    inputgui.start()