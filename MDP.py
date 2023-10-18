# CodeSkulptor runs Python programs in your browser.
# Click the upper left button to run this simple demo.

# CodeSkulptor is tested to run in recent versions of
# Chrome, Firefox, and Safari.

# Markov Decision Process Simulation
# Created by Abhishek Varghese


import random
from bisect import bisect
import numpy as np
import SimpleGUICS2Pygame.simpleguics2pygame as simplegui


# Helper Functions
def weighted_choice(data):
    values, weights = zip(*data)
    total = 0
    cum_weights = []
    for w in weights:
        total += w
        cum_weights.append(total)
    x = random.uniform(0, 1)
    i = bisect(cum_weights, x)
    return values[i]


# Building the environment
class MDP_env:
    def __init__(self) -> None:
        self.seed = 13
        random.seed(self.seed)
        # Board Values
        # 0 is empty
        # 1 is None right now
        # 2 is wall
        # 3 is hollow
        # 4 is goal
        self.board1 = [[0, 0, 0, 4], [0, 2, 0, 3], [0, 0, 0, 0]]
        self.start_pos1 = (2, 3)

        self.board2 = [
            [0, 0, 4, 0, 0],
            [2, 0, 2, 0, 3],
            [0, 0, 0, 3, 0],
            [0, 2, 0, 0, 0],
        ]
        self.start_pos2 = (3, 4)

        self.board = self.board1
        self.start_pos = self.start_pos1
        self.player_pos = self.start_pos

        # 0 is left
        # 1 is right
        # 2 is up
        # 3 is down
        # None is No action available
        self.action = [[0] * len(self.board[0]) for _ in range(len(self.board))]
        print(self.action)
        for i in range(len(self.action)):
            for j in range(len(self.action[i])):
                if self.board[i][j] == 0:
                    self.action[i][j] = random.choice([0, 1, 2, 3])
                else:
                    self.action[i][j] = None

        # +1 for goal, -1 for danger and -0.02 everywhere
        self.reward = [[-0.02] * len(self.board[0]) for _ in range(len(self.board))]
        for i in range(len(self.reward)):
            for j in range(len(self.reward[i])):
                if self.board[i][j] == 4:
                    self.reward[i][j] = 1
                elif self.board[i][j] == 3:
                    self.reward[i][j] = -1
                else:
                    self.reward[i][j] = -0.02

        # Action - consequence the actual prob distribution
        # which only environment knows
        self.action_consequence = dict()
        self.action_consequence[0] = [(3, 0.1), (0, 0.8), (2, 0.1)]
        self.action_consequence[1] = [(2, 0.1), (1, 0.8), (3, 0.1)]
        self.action_consequence[2] = [(0, 0.1), (2, 0.8), (1, 0.1)]
        self.action_consequence[3] = [(1, 0.1), (3, 0.8), (0, 0.1)]

        # Miscellaneous
        self.canvas_width = 700
        self.canvas_height = 500
        self.n_tiles_horiz = len(self.board[0])
        self.n_tiles_vertical = len(self.board)
        self.tile_size = self.canvas_height / self.n_tiles_vertical

        self.tile_dims = (self.tile_size, self.tile_size)
        self.arrow_size = self.tile_size / 6
        self.num_moves = 0
        self.type_move = "player"

        # Following are the parts of the agent's brain. Only to be used in estimate functions
        self.gamma = 0.99
        # Action - consequence estimates [action,# of times happened, #of times taken]
        self.action_consequence_estimates = dict()
        self.action_consequence_estimates[0] = {"ntimes": 3, 3: 1, 0: 1, 2: 1}
        self.action_consequence_estimates[1] = {"ntimes": 3, 2: 1, 1: 1, 3: 1}
        self.action_consequence_estimates[2] = {"ntimes": 3, 0: 1, 2: 1, 1: 1}
        self.action_consequence_estimates[3] = {"ntimes": 3, 1: 1, 3: 1, 0: 1}

        # Initialising the Values
        self.value_estimate = [[0] * len(self.board[0]) for _ in range(len(self.board))]
        self.value_estimate = np.array(self.value_estimate, dtype="float32")


class MDP_with_GUI:
    def __init__(self, frame, env: MDP_env):
        self.env = env
        frame.add_button("1 Step", self.one_step)
        frame.add_button("Single Run", self.single_run)
        frame.add_button("Play", self.play)
        frame.add_button("Stop", self.stop)
        frame.add_button("Reset", self.reset)
        frame.set_draw_handler(self.draw_all)

        print(self.env.action_consequence_estimates)
        frame.add_label("\n\n\n")
        frame.add_label("\n\n\n")
        frame.add_label(
            "\n\nProbability Distribution known by the agent for action left : \n\n"
        )
        frame.add_label("\n\n\n")
        self.label1 = frame.add_label(
            "P_left_goes-left = %.3f"
            % (
                self.env.action_consequence_estimates[0][0]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )
        self.label2 = frame.add_label(
            "P_left_goes-up = %.3f"
            % (
                self.env.action_consequence_estimates[0][2]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )
        self.label3 = frame.add_label(
            "P_left_goes-down = %.3f"
            % (
                self.env.action_consequence_estimates[0][3]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )
        frame.add_label("\n\n\n")
        frame.add_label("\n\n\n")
        frame.add_button("Board1", self.set_board1)
        frame.add_button("Board2", self.set_board2)
        self.timer_play = simplegui.create_timer(500, self.one_step)
        self.timer_singlerun = simplegui.create_timer(500, self.one_step)

    def flatten(self, i, j):
        return self.env.n_tiles_horiz * i + j

    # Handler to draw on canvas
    def draw_arrow(self, canvas, action):
        padding = self.env.tile_size / 12
        inner_tile_size = self.env.tile_size - 2 * padding
        for i in range(len(action)):
            curr_pos_vertical = self.env.tile_size * i + padding
            for j in range(len(action[i])):
                curr_pos_horizontal = self.env.tile_size * j + padding
                canvas.draw_text(
                    str(self.env.value_estimate[i][j]),
                    (curr_pos_horizontal, curr_pos_vertical + 1 * padding),
                    10,
                    "white",
                )
                if action[i][j] != None:
                    if action[i][j] == 2:
                        point_list = [
                            [curr_pos_horizontal, curr_pos_vertical + inner_tile_size],
                            [
                                curr_pos_horizontal + self.env.arrow_size / 2,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size,
                            ],
                            [
                                curr_pos_horizontal + self.env.arrow_size,
                                curr_pos_vertical + inner_tile_size,
                            ],
                        ]
                    elif action[i][j] == 0:
                        point_list = [
                            [
                                curr_pos_horizontal,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size / 2,
                            ],
                            [
                                curr_pos_horizontal + self.env.arrow_size,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size,
                            ],
                            [
                                curr_pos_horizontal + self.env.arrow_size,
                                curr_pos_vertical + inner_tile_size,
                            ],
                        ]
                    elif action[i][j] == 1:
                        point_list = [
                            [
                                curr_pos_horizontal,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size,
                            ],
                            [curr_pos_horizontal, curr_pos_vertical + inner_tile_size],
                            [
                                curr_pos_horizontal + self.env.arrow_size,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size / 2,
                            ],
                        ]
                    elif action[i][j] == 3:
                        point_list = [
                            [
                                curr_pos_horizontal,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size,
                            ],
                            [
                                curr_pos_horizontal + self.env.arrow_size,
                                curr_pos_vertical
                                + inner_tile_size
                                - self.env.arrow_size,
                            ],
                            [
                                curr_pos_horizontal + self.env.arrow_size / 2,
                                curr_pos_vertical + inner_tile_size,
                            ],
                        ]
                    canvas.draw_polygon(point_list, 1, "white", "white")

    def draw_board(self, canvas, board):
        for i in range(len(board)):
            curr_pos_vertical = self.env.tile_size * i
            for j in range(len(board[i])):
                fill_color = "black"
                if board[i][j] == 2:
                    fill_color = "grey"
                if board[i][j] == 3:
                    fill_color = "red"
                if board[i][j] == 4:
                    fill_color = "green"

                curr_pos_horizontal = self.env.tile_size * j
                point_list = [
                    [curr_pos_horizontal, curr_pos_vertical],
                    [curr_pos_horizontal + self.env.tile_size, curr_pos_vertical],
                    [
                        curr_pos_horizontal + self.env.tile_size,
                        curr_pos_vertical + self.env.tile_size,
                    ],
                    [curr_pos_horizontal, curr_pos_vertical + self.env.tile_size],
                ]
                canvas.draw_polygon(point_list, 10, "blue", fill_color)

        canvas.draw_circle(
            (
                self.env.player_pos[1] * self.env.tile_size + self.env.tile_size / 2,
                self.env.player_pos[0] * self.env.tile_size + self.env.tile_size / 2,
            ),
            self.env.tile_size / 4,
            2,
            "yellow",
            "yellow",
        )

    def draw_all(self, canvas):
        self.draw_board(canvas, self.env.board)
        self.draw_arrow(canvas, self.env.action)

    # Handler for mouse click
    def set_board1(self):
        self.env.board = self.env.board1
        self.env.start_pos = self.env.start_pos1
        self.reset()

    def set_board2(self):
        self.env.board = self.env.board2
        self.env.start_pos = self.env.start_pos2
        self.reset()

    def reset(self):
        self.env.player_pos = self.env.start_pos
        self.env.seed += 1
        random.seed(self.env.seed)
        # Reser action consequence estimates
        self.env.action_consequence_estimates = dict()
        self.env.action_consequence_estimates[0] = {"ntimes": 3, 3: 1, 0: 1, 2: 1}
        self.env.action_consequence_estimates[1] = {"ntimes": 3, 2: 1, 1: 1, 3: 1}
        self.env.action_consequence_estimates[2] = {"ntimes": 3, 0: 1, 2: 1, 1: 1}
        self.env.action_consequence_estimates[3] = {"ntimes": 3, 1: 1, 3: 1, 0: 1}
        self.label1.set_text(
            "P_left_goes-left = %.3f"
            % (
                self.env.action_consequence_estimates[0][0]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )
        self.label2.set_text(
            "P_left_goes-up = %.3f"
            % (
                self.env.action_consequence_estimates[0][2]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )
        self.label3.set_text(
            "P_left_goes-down = %.3f"
            % (
                self.env.action_consequence_estimates[0][3]
                / self.env.action_consequence_estimates[0]["ntimes"]
            )
        )

        # reset value_estimates
        self.env.value_estimate = [
            [0] * len(self.env.board[0]) for _ in range(len(self.env.board))
        ]
        self.env.value_estimate = np.array(self.env.value_estimate, dtype="float32")

        # reset action to random
        self.env.action = [
            [0] * len(self.env.board[0]) for _ in range(len(self.env.board))
        ]
        for i in range(len(self.env.action)):
            for j in range(len(self.env.action[i])):
                if self.env.board[i][j] == 0:
                    self.env.action[i][j] = random.choice([0, 1, 2, 3])
                else:
                    self.env.action[i][j] = None

        # Stop all timers
        self.timer_play.stop()
        self.timer_singlerun.stop()

        # reset reward
        self.env.reward = [
            [-0.02] * len(self.env.board[0]) for _ in range(len(self.env.board))
        ]
        for i in range(len(self.env.reward)):
            for j in range(len(self.env.reward[i])):
                if self.env.board[i][j] == 4:
                    self.env.reward[i][j] = 1
                elif self.env.board[i][j] == 3:
                    self.env.reward[i][j] = -1
                else:
                    self.env.reward[i][j] = -0.02

        self.env.n_tiles_horiz = len(self.env.board[0])
        self.env.n_tiles_vertical = len(self.env.board)
        self.env.tile_size = self.env.canvas_height / self.env.n_tiles_vertical

        self.env.tile_dims = (self.env.tile_size, self.env.tile_size)
        self.env.arrow_size = self.env.tile_size / 6
        self.env.type_move = "player"

    def one_step(self):
        self.env.num_moves += 1
        if self.env.type_move == "player":
            wandering = self.agent_move()
            if not wandering or self.env.num_moves > 10:
                self.env.type_move = "computation"
        elif self.env.type_move == "computation":
            if self.timer_play.is_running():
                self.timer_play.stop()
                self.category = "play"
            elif self.timer_singlerun.is_running():
                self.timer_singlerun.stop()
                self.category = "singlerun"
            else:
                self.category = None
            self.timer_singlerun.stop()
            self.label1.set_text(
                "P_left_goes-left = %.3f"
                % (
                    self.env.action_consequence_estimates[0][0]
                    / self.env.action_consequence_estimates[0]["ntimes"]
                )
            )
            self.label2.set_text(
                "P_left_goes-up = %.3f"
                % (
                    self.env.action_consequence_estimates[0][2]
                    / self.env.action_consequence_estimates[0]["ntimes"]
                )
            )
            self.label3.set_text(
                "P_left_goes-down = %.3f"
                % (
                    self.env.action_consequence_estimates[0][3]
                    / self.env.action_consequence_estimates[0]["ntimes"]
                )
            )
            self.policy_iteration()
            self.env.type_move = "blank"
            if self.category == "play":
                self.timer_play.start()
            elif self.category == "singlerun":
                self.timer_singlerun.start()
        else:
            self.env.type_move = "player"
            self.env.player_pos = self.env.start_pos
            self.env.num_moves = 0
            self.timer_singlerun.stop()

    def single_run(self):
        self.timer_play.stop()
        self.timer_singlerun.start()

    def play(self):
        self.timer_singlerun.stop()
        self.timer_play.start()

    def stop(self):
        self.timer_singlerun.stop()
        self.timer_play.stop()

    def agent_move(self):
        print(len(self.env.action), len(self.env.action[1]))
        sampled_action = weighted_choice(
            self.env.action_consequence[
                self.env.action[self.env.player_pos[0]][self.env.player_pos[1]]
            ]
        )
        # Agent Updating what has happened in his mind
        self.env.action_consequence_estimates[
            self.env.action[self.env.player_pos[0]][self.env.player_pos[1]]
        ][sampled_action] += 1
        self.env.action_consequence_estimates[
            self.env.action[self.env.player_pos[0]][self.env.player_pos[1]]
        ]["ntimes"] += 1

        if sampled_action == 0:
            new_pos = (self.env.player_pos[0], self.env.player_pos[1] - 1)
            if new_pos[1] < 0 or self.env.board[new_pos[0]][new_pos[1]] == 2:
                pass
            else:
                self.env.player_pos = new_pos
        elif sampled_action == 1:
            new_pos = (self.env.player_pos[0], self.env.player_pos[1] + 1)
            if (
                new_pos[1] >= self.env.n_tiles_horiz
                or self.env.board[new_pos[0]][new_pos[1]] == 2
            ):
                pass
            else:
                self.env.player_pos = new_pos
        elif sampled_action == 2:
            new_pos = (self.env.player_pos[0] - 1, self.env.player_pos[1])
            if new_pos[0] < 0 or self.env.board[new_pos[0]][new_pos[1]] == 2:
                pass
            else:
                self.env.player_pos = new_pos
        elif sampled_action == 3:
            new_pos = (self.env.player_pos[0] + 1, self.env.player_pos[1])
            if (
                new_pos[0] >= self.env.n_tiles_vertical
                or self.env.board[new_pos[0]][new_pos[1]] == 2
            ):
                pass
            else:
                self.env.player_pos = new_pos

        if self.env.reward[self.env.player_pos[0]][self.env.player_pos[1]] in (1, -1):
            return False

        return True

    def policy_iteration_onestep(self):
        A = np.zeros(
            (
                self.env.n_tiles_horiz * self.env.n_tiles_vertical,
                self.env.n_tiles_horiz * self.env.n_tiles_vertical,
            ),
            dtype="float32",
        )
        for i in range(len(self.env.value_estimate)):
            for j in range(len(self.env.value_estimate[i])):
                A[self.flatten(i, j), self.flatten(i, j)] = 1
                if self.env.action[i][j] == None:
                    continue
                possible_actions = self.env.action_consequence_estimates[
                    self.env.action[i][j]
                ]
                for act in possible_actions:
                    if act == "ntimes":
                        continue
                    prob = possible_actions[act] / possible_actions["ntimes"]
                    if act == 0:
                        if j != 0 and self.env.board[i][j - 1] != 2:
                            A[self.flatten(i, j), self.flatten(i, j - 1)] += (
                                -prob * self.env.gamma
                            )
                        else:
                            A[self.flatten(i, j), self.flatten(i, j)] += (
                                -prob * self.env.gamma
                            )
                    elif act == 1:
                        if (
                            j != len(self.env.value_estimate[i]) - 1
                            and self.env.board[i][j + 1] != 2
                        ):
                            A[self.flatten(i, j), self.flatten(i, j + 1)] += (
                                -prob * self.env.gamma
                            )
                        else:
                            A[self.flatten(i, j), self.flatten(i, j)] += (
                                -prob * self.env.gamma
                            )
                    elif act == 2:
                        if i != 0 and self.env.board[i - 1][j] != 2:
                            A[self.flatten(i, j), self.flatten(i - 1, j)] += (
                                -prob * self.env.gamma
                            )
                        else:
                            A[self.flatten(i, j), self.flatten(i, j)] += (
                                -prob * self.env.gamma
                            )
                    elif act == 3:
                        if (
                            i != len(self.env.value_estimate) - 1
                            and self.env.board[i + 1][j] != 2
                        ):
                            A[self.flatten(i, j), self.flatten(i + 1, j)] += (
                                -prob * self.env.gamma
                            )
                        else:
                            A[self.flatten(i, j), self.flatten(i, j)] += (
                                -prob * self.env.gamma
                            )

        B = np.array(self.env.reward).flatten()
        X = np.linalg.inv(A).dot(B)
        for i in range(len(self.env.value_estimate)):
            for j in range(len(self.env.value_estimate[i])):
                if self.env.board[i][j] != 2:
                    self.env.value_estimate[i][j] = X[self.flatten(i, j)]

        print("Old Action : ", self.env.action)
        for i in range(len(self.env.action)):
            for j in range(len(self.env.action[i])):
                if self.env.action[i][j] == None:
                    continue
                to_take_max = []
                for each_act in [0, 1, 2, 3]:
                    if (
                        (each_act == 0 and (j == 0 or self.env.board[i][j - 1] == 2))
                        or (
                            each_act == 1
                            and (
                                j == len(self.env.value_estimate[i]) - 1
                                or self.env.board[i][j + 1] == 2
                            )
                        )
                        or (each_act == 2 and (i == 0 or self.env.board[i - 1][j] == 2))
                        or (
                            each_act == 3
                            and (
                                i == len(self.env.value_estimate) - 1
                                or self.env.board[i + 1][j] == 2
                            )
                        )
                    ):
                        to_take_max.append(-20)
                        continue
                    possible_actions = self.env.action_consequence_estimates[each_act]
                    overall_sum = 0
                    for act in possible_actions:
                        if act == "ntimes":
                            continue
                        prob = possible_actions[act] / possible_actions["ntimes"]
                        if act == 0:
                            if j != 0 and self.env.board[i][j - 1] != 2:
                                overall_sum += prob * self.env.value_estimate[i][j - 1]
                            else:
                                overall_sum += prob * self.env.value_estimate[i][j]
                        elif act == 1:
                            if (
                                j != len(self.env.value_estimate[i]) - 1
                                and self.env.board[i][j + 1] != 2
                            ):
                                overall_sum += prob * self.env.value_estimate[i][j + 1]
                            else:
                                overall_sum += prob * self.env.value_estimate[i][j]
                        elif act == 2:
                            if i != 0 and self.env.board[i - 1][j] != 2:
                                overall_sum += prob * self.env.value_estimate[i - 1][j]
                            else:
                                overall_sum += prob * self.env.value_estimate[i][j]
                        elif act == 3:
                            if (
                                i != len(self.env.value_estimate) - 1
                                and self.env.board[i + 1][j] != 2
                            ):
                                overall_sum += prob * self.env.value_estimate[i + 1][j]
                            else:
                                overall_sum += prob * self.env.value_estimate[i][j]
                    to_take_max.append(overall_sum)

                print("i : ", i, "j : ", j, "list = ", to_take_max)
                self.env.action[i][j] = np.argmax(to_take_max)
        print("New Action List : ", self.env.action)

    def policy_iteration(self):
        for i in range(100):
            self.policy_iteration_onestep()


if __name__ == "__main__":
    mdp_env = MDP_env()
    frame = simplegui.create_frame(
        "Home", mdp_env.canvas_width, mdp_env.canvas_height, 400
    )
    mdp_gui = MDP_with_GUI(frame, mdp_env)
    # Start the frame animation
    frame.start()
