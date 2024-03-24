import json
import os
import time



path = 'training_data/bhav_episode_0.json'

data = json.load(open(path, 'r'))
for frame in data:
    os.system('clear')
    print(frame['grid'], end='\r')
    time.sleep(0.05)
