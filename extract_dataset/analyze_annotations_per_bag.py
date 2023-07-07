#!/usr/bin/env python3
import numpy as np
import os
from matplotlib import pyplot as plt


load_directory = '/usr/local/data/kvirji/offroad_navigation_dataset/annotations/'
save_directory = '/usr/local/data/kvirji/offroad_autonomous_navigation/extract_dataset/plots/'
labels = ['Tree', 'Other Obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable Grass', 'Smooth Road', 'Wet Leaves']
bag_dist = {}


for filename in os.listdir(load_directory):
    f = os.path.join(load_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        bag = filename[6:-9]
        with open(f) as file:
            label = int(file.readline())
            if bag in bag_dist.keys():
                bag_dist[bag][label] += 1
            else:
                bag_dist[bag] = np.zeros(len(labels))
                bag_dist[bag][label] += 1

for key, value in bag_dist.items():
    plt.plot(value, label=key, marker='.')
    break
plt.legend()
plt.show()




