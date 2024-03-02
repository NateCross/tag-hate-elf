import warnings
warnings.filterwarnings('ignore')
from transformers import logging
logging.set_verbosity_error()

import PySimpleGUI as sg
import joblib
from concurrent.futures import ThreadPoolExecutor

def load_ensemble(filename: str, window: sg.Window):
    try:
        ensemble = joblib.load('ensemble-hard.pkl')
    except FileNotFoundError:
        sg.PopupError(
            'ERROR: "ensemble-hard.pkl" not found. Please train a hard voting ensemble first.'
        )
        return None
    return ensemble

def predict(ensemble, text: str):
    pass
    


def hard_voting():
    # ensemble_window = sg.Window()
    loading_popup = sg.PopupNoButtons(
        'Loading hard voting ensemble...,'
    )
    # window.start_thread()


    # print(ensemble)

def soft_voting():
    pass

def stacking():
    pass

def event_loop(window: sg.Window):
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
            break
        if event == 'Hard Voting':
            hard_voting()

def main_window():
    button_frame = sg.Frame(
        'Select an Ensemble below:',
        layout=[
            [           
                sg.Button(
                    'Hard Voting',
                ),
            ],
            [
                sg.Button(
                    'Soft Voting',
                ),
            ],
            [
                sg.Button(
                    'Stacking',
                ),
            ],
        ],
        element_justification='c',
        
    )

    layout = [
        [
            sg.Text(
                'TAG-HATE-ELF',
                justification='c',
            ),
        ],
        [button_frame],
        [
            sg.Exit(),
        ],
    ]

    # Create the Window
    window = sg.Window(
        'TAG-HATE-ELF', 
        layout,
        element_justification='c',
    )

    event_loop(window)

    window.close()


if __name__ == "__main__":
    main_window()
