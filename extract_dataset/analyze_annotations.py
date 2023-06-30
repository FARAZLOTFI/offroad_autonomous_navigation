#!/usr/bin/env python3
import numpy as np
import os
from matplotlib import pyplot as plt
import math

load_directory = '/home/farnoosh/rc_car/annotations_khalil/'
save_directory = '/home/farnoosh/rc_car/src/extract_dataset/plots/'
labels = ['Tree', 'Other Obstacles', 'Human', 'Waterhole', 'Mud', 'Jump', 'Traversable Grass', 'Smooth Road', 'Wet Leaves']

label_count = np.zeros(len(labels))
for filename in os.listdir(load_directory):
    f = os.path.join(load_directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f) as file:
            label = int(file.readline())
            label_count[label] += 1

plt.figure(figsize=(15, 6))
plt.bar(labels, label_count)
plt.xlabel('Count')
plt.title('Label distribution')

# plt.show()
plt.savefig(save_directory + 'label_dist.png')
