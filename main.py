import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI
from value_it import ValueIterationGUI
from qlearning import Qlearning_with_GUI

if __name__ == "__main__":
    frame = simplegui.create_frame("MDP visualisation", 700,600)
    value_it_gui = ValueIterationGUI(frame)
    qlearn_gui = Qlearning_with_GUI(frame)
    send_fn_dict = {
        "value_iteration": value_it_gui.take_over,
        "q_learning": qlearn_gui.take_over,
    }
    inputgui = MDPGUI(frame, send_board_data_to=send_fn_dict)
    inputgui.start()