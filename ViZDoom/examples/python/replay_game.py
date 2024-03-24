from vizdoom import DoomGame, Mode
#from vizdoom.opencv import ScreenResolutionManager
#import cv2
import os
from pynput import keyboard
import time
import numpy as np
import json

game = DoomGame()
#game.load_config("basic.cfg")
game.load_config(os.path.join('/Users/bhav/experiments/mistral-hackathon/repos/GameCopilot/ViZDoom/scenarios', "basic.cfg"))
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.set_labels_buffer_enabled(True)
game.set_render_hud(False)
game.init()

#resolution_manager = ScreenResolutionManager()
#resolution_manager.set_game(game)
#resolution_manager.set_mode(Mode.ASYNC_PLAYER)

def generate_ascii_grid(bounding_boxes, wall_buffer, floor_buffer, screen_width, screen_height, grid_width=64, grid_height=32):
    # Normalize screen dimensions to 32x32 grid
    scale_x = 1.0*grid_width / screen_width
    scale_y = 1.0*grid_height / screen_height

    # Create a 32x32 grid filled with spaces
    grid = [[' ' for _ in range(grid_width)] for _ in range(grid_height)]

    for i in range(grid_height):
        for j in range(grid_width):
            x1 = int(j / scale_x)
            y1 = int(i / scale_y)
            x2 = int((j + 1) / scale_x)
            y2 = int((i + 1) / scale_y)

            area = (y2 - y1) * (x2 - x1)

            if area > 0:
                wall_score = sum(wall_buffer[y1:y2, x1:x2].flatten()) / area
                floor_score = sum(floor_buffer[y1:y2, x1:x2].flatten()) / area
            else:
                wall_score = 0
                floor_score = 0

            if wall_score > 0.5:
                grid[i][j] = 'W'
            elif floor_score > 0.5:
                grid[i][j] = 'F'

    # Iterate over the bounding boxes
    for x, y, w, h, label in bounding_boxes:
        # Normalize coordinates and dimensions to 32x32 grid
        x_norm = int(x * scale_x)
        y_norm = int(y * scale_y)
        w_norm = int(w * scale_x)
        h_norm = int(h * scale_y)

        # Ensure coordinates and dimensions are within the grid
        x_norm = max(0, min(grid_width-1, x_norm))
        y_norm = max(0, min(grid_height-1, y_norm))
        w_norm = max(0, min(grid_width - x_norm, w_norm))
        h_norm = max(0, min(grid_height - y_norm, h_norm))

        # Draw the bounding box on the grid
        for i in range(h_norm):
            for j in range(w_norm):
                grid[y_norm + i][x_norm + j] = label

    # Convert the grid to a string
    ascii_grid = ''
    for row in grid:
        ascii_grid += ''.join(row) + '\n'

    return ascii_grid


def get_object_name_char(object_name):
    if object_name == 'Cacodemon':
        return 'E'
    if object_name == 'DoomPlayer':
        return 'P'
    else:
        return object_name[0]

def convert_labels_to_representation(labels, wall_buffer, floor_buffer, screen_height=320, screen_width=240):
    reps = []
    #screen_height = 320
    #screen_width = 240
    for label in labels:
        #rep = (32*label.x/screen_width, 32*label.y/screen_height, 32*label.width/screen_width, 32*label.height/screen_height, label.object_name[0])
        rep = (label.x, label.y, label.width, label.height, get_object_name_char(label.object_name))
        #rep = (label.y, label.x, label.height, label.width, label.object_name[0])
        print(label.object_name)
        reps.append(rep)
    grid = generate_ascii_grid(reps, wall_buffer, floor_buffer, screen_height, screen_width)
    return grid

def one_hot(i, max_num=7):
    arr = [False for _ in range(max_num)]
    arr[i] = True
    return arr
'''
key_mappings = {
    keyboard.Key.left: Button.TURN_LEFT,
    keyboard.Key.right: Button.TURN_RIGHT,
    keyboard.Key.up: Button.MOVE_FORWARD,
    keyboard.Key.down: Button.MOVE_BACKWARD,
    'a': Button.MOVE_LEFT,
    'd': Button.MOVE_RIGHT,
    keyboard.Key.space: Button.ATTACK
}
'''
available_actions = [
                'MOVE_FORWARD',
                'TURN_LEFT',
                'TURN_RIGHT',
                'MOVE_BACKWARD',
                'MOVE_LEFT',
                'MOVE_RIGHT',
                'ATTACK'
]
key_mappings = {
    keyboard.Key.up: 0,
    keyboard.Key.left: 1,
    keyboard.Key.right: 2,
    keyboard.Key.down: 3,
    'a': 4,
    'd': 5,
    keyboard.Key.space: 6,
    keyboard.Key.esc: 7,
}

action_mappings = {
    0: one_hot(0),   # Move forward
    1: one_hot(1),   # Turn left
    2: one_hot(2),    # Turn right
    3: one_hot(3),    # Move backward 
    4: one_hot(4),    # Move left 
    5: one_hot(5),    # Move right
    6: one_hot(6),    # Attack 
    7: None,
}

pressed_keys = set()

#def on_press(key):
#    if key in key_mappings:
#        pressed_keys.add(key_mappings[key])

#def on_release(key):
#    if key in key_mappings:
#        pressed_keys.discard(key_mappings[key])

#listener = keyboard.Listener(on_press=on_press, on_release=on_release)
#listener.start()

def llm_call(last_state):
    return None

episodes = 10
screen_height = 320
screen_width = 240
wall_id = 0
floor_id = 1

all_recorded_actions = []
episode_file = '/Users/bhav/experiments/mistral-hackathon/repos/GameCopilot/ViZDoom/training_data/bhav_episode_0.json'
recorded_episode = json.load(open(episode_file, 'r'))
for step in recorded_episode:
    all_recorded_actions.append(step['action'])

def get_next_action():
    if len(all_recorded_actions) == 0:
        return None
    action = all_recorded_actions.pop(0)
    return action

for episode in range(episodes):
    game.new_episode()
    episode_data = []
    all_labels = {}
    next_action = get_next_action()#None
    while not game.is_episode_finished():
        state = game.get_state()
        #if state.tic < 20:
        #    continue

        # Get user input for action
        #action_input = input("Enter action (0-3): ")
        #if len(pressed_keys) == 0:
        #    continue
        #pressed_key = list(pressed_keys)[0]
        #print('Pressed key: ', pressed_key)
        try:
            #action_index = pressed_key#int(key_mappings[pressed_key])
            #if action_index == 7:
            #    # End game
            #    break
            if next_action is None:
                #action = action_mappings[0]
                continue
            else:
                action = next_action
            #action = action_mappings.get(action_index, [False, False, False])

            wall_buffer = np.zeros_like(state.labels_buffer)
            floor_buffer = np.zeros_like(state.labels_buffer)
            wall_buffer[state.labels_buffer == wall_id] = 1
            floor_buffer[state.labels_buffer == floor_id] = 1

            print(state.labels)
            for label in state.labels:
                all_labels[label.object_name] = 0
            grid = convert_labels_to_representation(state.labels, wall_buffer, floor_buffer, screen_height=320, screen_width=240)
            print(grid)
            example = {}
            example['grid_height'] = 32
            example['grid_width'] = 32
            example['screen_width'] = screen_width
            example['screen_height'] = screen_height
            example['grid'] = grid
            example['available_actions'] = available_actions
            example['action'] = action
            example['killcount'] = state.game_variables[0]
            example['health'] = state.game_variables[1]
            example['armor'] = state.game_variables[2]
            example['ammo2'] = state.game_variables[3]
            reward = game.make_action(action)
            example['reward'] = reward
            episode_data.append(example)
            # TODO: Make api call to LLM.
            next_action = get_next_action()
            print(next_action)

        except ValueError:
            print("Invalid input. Using random action.")
            #action = choice([0, 1, 2, 3])

        print(f"State: {state.number}, Action: {action}, Reward: {reward}")

        # Capture and save the frame
        #frame = resolution_manager.grab_screen()
        #cv2.imwrite(f"frames/frame_{state.number}.png", frame)

        fp = open(f'./training_data/episode_{episode}.json', 'w')
        json.dump(episode_data, fp, indent=4)
        fp.close()

game.close()
