#!/usr/bin/env python3

#####################################################################
# This script presents how to use Doom's native demo mechanism to
# replay episodes with perfect accuracy.
#####################################################################

import os
from random import choice
import json
import numpy as np

import vizdoom as vzd
from vizdoom import AutomapMode, DoomGame, Mode, ScreenFormat, ScreenResolution


game = vzd.DoomGame()

# Use other config file if you wish.
#game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
game.load_config(os.path.join('/Users/bhav/experiments/mistral-hackathon/repos/GameCopilot/ViZDoom/scenarios', "basic.cfg"))
game.set_episode_timeout(100)

# Record episodes while playing in 320x240 resolution without HUD
game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
game.set_render_hud(False)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.PLAYER)

game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)
game.set_automap_mode(AutomapMode.OBJECTS_WITH_SIZE)
game.init()

actions = [[True, False, False], [False, True, False], [False, False, True]]


# Run and record this many episodes
episodes = 2

def generate_ascii_grid(bounding_boxes, wall_buffer, floor_buffer, screen_width, screen_height):
    # Normalize screen dimensions to 32x32 grid
    scale_x = 32 / screen_width
    scale_y = 32 / screen_height

    # Create a 32x32 grid filled with spaces
    grid = [[' ' for _ in range(32)] for _ in range(32)]

    for i in range(32):
        for j in range(32):
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
        x_norm = max(0, min(31, x_norm))
        y_norm = max(0, min(31, y_norm))
        w_norm = max(0, min(32 - x_norm, w_norm))
        h_norm = max(0, min(32 - y_norm, h_norm))

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
 
# Recording
print("\nRECORDING EPISODES")
print("************************\n")

screen_height = 320
screen_width = 240

available_actions = ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK']
wall_id = 0
floor_id = 1
for i in range(episodes):

    # new_episode can record the episode using Doom's demo recording functionality to given file.
    # Recorded episodes can be reconstructed with perfect accuracy using different rendering settings.
    # This can not be used to record episodes in multiplayer mode.
    game.new_episode(f"episode{i}_rec.lmp")

    episode_data = []
    while not game.is_episode_finished():
        s = game.get_state()
        a = choice(actions)
        r = game.make_action(choice(actions))

        print(s.labels_buffer.shape)
        print(s.labels_buffer)
        wall_buffer = np.zeros_like(s.labels_buffer)
        floor_buffer = np.zeros_like(s.labels_buffer)
        wall_buffer[s.labels_buffer == wall_id] = 1
        floor_buffer[s.labels_buffer == floor_id] = 1

        print(s.labels)
        grid = convert_labels_to_representation(s.labels, wall_buffer, floor_buffer, screen_height=320, screen_width=240)
        print(grid)
        example = {}
        example['grid_height'] = 32
        example['grid_width'] = 32
        example['screen_width'] = screen_width
        example['screen_height'] = screen_height
        example['grid'] = grid
        example['available_actions'] = available_actions
        example['action'] = a
        example['killcount'] = s.game_variables[0]
        example['health'] = s.game_variables[1]
        example['armor'] = s.game_variables[2]
        example['ammo2'] = s.game_variables[3]
        episode_data.append(example)

        print(f"State #{s.number}")
        print("Action:", a)
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print(f"Episode {i} finished. Saved to file episode{i}_rec.lmp")
    print("Total reward:", game.get_total_reward())
    print("************************\n")
    fp = open(f'./training_data/episode_{i}.json', 'w')
    json.dump(episode_data, fp, indent=4)

game.new_episode()  # This is currently required to stop and save the previous recording.
game.close()
exit()

# New render settings for replay
game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
game.set_render_hud(True)

# Replay can be played in any mode.
game.set_mode(vzd.Mode.SPECTATOR)

game.set_labels_buffer_enabled(True)
game.init()

print("\nREPLAY OF EPISODE")
print("************************\n")

screen_height = 800 
screen_width = 600 
for i in range(episodes):

    # Replays episodes stored in given file. Sending game command will interrupt playback.
    game.replay_episode(f"episode{i}_rec.lmp")

    episode_data = []
    while not game.is_episode_finished():
        # Get a state
        s = game.get_state()
        grid = convert_labels_to_representation(s.labels, wall_buffer, floor_buffer, screen_height=800, screen_width=600)
        print(grid)
        example = {}
        example['grid'] = grid
        example['action'] = a
        example['killcount'] = s.game_variables[0]
        example['health'] = s.game_variables[1]
        example['armor'] = s.game_variables[2]
        example['ammo2'] = s.game_variables[3]
        episode_data.append(example)

        # Use advance_action instead of make_action to proceed
        game.advance_action()

        # Retrieve the last actions and the reward
        a = game.get_last_action()
        r = game.get_last_reward()

        print(f"State #{s.number}")
        print("Action:", a)
        print("Game variables:", s.game_variables[0])
        print("Reward:", r)
        print("=====================")

    print("Episode", i, "finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")

game.close()

# Delete recordings (*.lmp files).
for i in range(episodes):
    os.remove(f"episode{i}_rec.lmp")
