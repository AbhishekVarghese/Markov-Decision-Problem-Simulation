import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
from input_mdp import MDPGUI
from value_it import ValueIterationGUI
from qlearning import Qlearning_with_GUI

if __name__ == "__main__":
    frame = simplegui.create_frame("MDP visualisation", 700,600)
    
    inputgui = MDPGUI(frame)
    value_it_gui = ValueIterationGUI(frame)
    qlearn_gui = Qlearning_with_GUI(frame)
    
    send_fn_dict = {
        "input": inputgui.take_over,
        "value_iteration": value_it_gui.take_over,
        "q_learning": qlearn_gui.take_over,
    }
    inputgui.set_control_transfer(send_fn_dict)
    value_it_gui.set_control_transfer(send_fn_dict)
    qlearn_gui.set_control_transfer(send_fn_dict)

    inputgui.start()