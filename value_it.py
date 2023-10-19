import numpy as np
import random 
import itertools
import warnings
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI,Board
import math
import time

class ValueIteration():
    def __init__(self, rewards, wall_reward=-10):
        self.rewards = rewards
        self.wall_reward = wall_reward
        self.iteration2values = []
        self.current_iter = 0

    def run(self, max_iters, discount, reset=False):
        self.max_iters = max_iters
        self.current_iter = 0
        min_possible = np.min(self.rewards) - 100

        values = self.rewards.copy()
        values[values == self.wall_reward] = min_possible

        if reset:
            self.iteration2values = []

        wall_mask = self.rewards == self.wall_reward


        if len(self.iteration2values) == 0:
            self.iteration2values.append(values)
        for i in range(1, max_iters+1):
            if i < len(self.iteration2values):
                continue
            self.current_iter = i 

            prev_values = self.iteration2values[i-1]
            prev_left = discount * np.pad(
                prev_values[:, 1:], 
                pad_width=((0, 0), (0, 1)),
                constant_values=min_possible
            )
            prev_right = discount * np.pad(
                prev_values[:, :-1], 
                pad_width=((0, 0), (1, 0)),
                constant_values=min_possible
            )
            prev_up = discount * np.pad(
                prev_values[1:, :], 
                pad_width=((0, 1), (0, 0)),
                constant_values=min_possible
            )
            prev_down = discount * np.pad(
                prev_values[:-1, :], 
                pad_width=((1, 0), (0, 0)),
                constant_values=min_possible
            )
            curr_values = np.stack(
                (prev_values, prev_left, prev_right, prev_up, prev_down),
                axis=0
            )
            # print(curr_values.shape)
            curr_values = curr_values.max(axis=0)
            curr_values = curr_values * (1 - wall_mask)
            # print(curr_values.shape)
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
        self.speed_increment = 0.1
        self.timer_multiplier = 50

        self.cmap = {
            0: "black",
            1: "green",
            -1: "red",
            -10: "blue",
            "other": "cyan"
        }
        self.grid_color = "blue"
        self.grid_width = 2
        self.draw_mode = 0
        self.canvas_width = self.frame._canvas._width
        self.canvas_height = self.frame._canvas._height

    def take_over(self, board, start_pos):
        self.rewards = board
        self.board = board
        self.curr_pos = start_pos
        self.x_pad, self.y_pad, self.l = self.get_pad_l()

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

        self.frame.add_button("Show Policies", self.show_policies, width = 100)

        self.frame.add_label("\n"*10)

        self.frame.add_button("Show Agent Path", self.show_agent_path, width = 100)

        self.algorithm = ValueIteration(
            self.rewards,
        )
        self.algorithm.run(self.iteration, self.discount)
        self.update_values()

        self.draw_status = None
        self.frame.set_draw_handler(self.draw_board)



    def update_values(self, iteration=None):
        if iteration is None:
            iteration = self.iteration
        self.values = self.algorithm.iteration2values[iteration]
        mx, mn = self.values.max(), self.values.min()
        self.values = (self.values - mn) / (mx - mn) if mx != mn else np.ones(self.values.shape)

    def update_discount(self, mode):
        def common(x):
            assert (x >= 0) and (x <= 1), "Error: Discount going out of bounds"
            self.discount = x
            self.d_label.set_text("{:.3f}".format(self.discount))
            self.algorithm.run(self.iteration, self.discount, reset=True)
            self.update_values()

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
            assert (x >= 0), "Error: Iteration has to be >=0"
            self.iteration = x
            self.it_label.set_text(str(self.iteration))
            if self.iteration > self.algorithm.max_iters:
                new_max_iter = self.iteration//self.maxiter_increment * (self.maxiter_increment + 1)
                self.algorithm.run(new_max_iter, self.discount)
            self.update_values()

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
                x = int(float(x))
                common(x)
            return handler
        elif mode == "+" or mode == "-":
            delta = self.speed_increment if mode == "+" else -self.speed_increment
            def handler():
                x = self.timer_speed + delta
                common(x)
            return handler


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
        
    def show_policies(self):
        pass

    def show_agent_path(self):
        pass

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
                if color == "black" :
                    value_color = "rgb({0}, {0}, {0})".format(int(255*self.values[i, j]))
                    canvas.draw_polygon(
                        rect, self.grid_width, 
                        self.grid_color, 
                        value_color
                    )
                else :
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
        i, j  = self.curr_pos
        canvas.draw_circle(
            self.ij2xy(i+0.5, j+0.5), 
            self.l//4, 2, 
            "yellow", "yellow"
        )
        if self.draw_status is not None:
            self.frame._canvas.draw_text(
                self.draw_status,
                (20, 30),
                font_size=20,
                font_color="white"
            )


if __name__ == "__main__" :
    frame = simplegui.create_frame("Value Iteration", 700,500)
    value_it_gui = ValueIterationGUI(frame)
    inputgui = MDPGUI(frame, send_board_data_to=value_it_gui.take_over)
    inputgui.start()