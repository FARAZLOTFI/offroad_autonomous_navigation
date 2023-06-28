#!/usr/bin/env python3
import numpy as np
import os
from matplotlib import pyplot as plt
import math


def euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]  # in radians


directory = '/home/farnoosh/rc_car/topic_data/'
bins = 50

data_matrix = []
euler_angle_matrix = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'rb') as datafile:
            data = np.load(datafile)
            data_matrix.append(data)
            euler_angle_matrix.append(euler_from_quaternion(data[2], data[3], data[4], data[5]))

data_matrix = np.asarray(data_matrix)
euler_angle_matrix = np.asarray(euler_angle_matrix)

plt.hist(data_matrix[:, 0], bins=bins)
plt.title('Steering Angle')
plt.ylabel('Count')
plt.savefig('steering_angle_dist.png')

plt.clf()

plt.hist(data_matrix[:, 1], bins=bins)
plt.title('Throttle')
plt.ylabel('Count')
plt.savefig('throttle_dist.png')

plt.clf()

max_stamps = np.max(data_matrix[:, 8:12], axis=1)
min_stamps = np.min(data_matrix[:, 8:12], axis=1)
max_diff = max_stamps - min_stamps
plt.hist(max_diff, bins=bins)
plt.title('Maximum Time Delay Between Synchronized Signals')
plt.ylabel('Count')
plt.xlabel('Seconds')
plt.savefig('delay_dist.png')

plt.clf()

plt.hist(euler_angle_matrix[:, 2], bins=bins)
plt.title('Heading angle - yaw')
plt.ylabel('Count')
plt.savefig('heading_angle_yaw.png')




