#!/usr/bin/env python3
import numpy as np
import cv2

FILE_NUMBER = 1126
BAG_NAME = '2023-06-13-10-32-59'

with open('/usr/local/data/kvirji/offroad_navigation_dataset/topics/topics_{bag_name}_{file_number:04d}.npy'.format(bag_name= BAG_NAME, file_number=FILE_NUMBER), 'rb') as datafile:
    data = np.load(datafile)

rgb_image = cv2.imread('/usr/local/data/kvirji/offroad_navigation_dataset/images/image_{bag_name}_{file_number:04d}.jpg'.format(bag_name= BAG_NAME, file_number=FILE_NUMBER))

with open('/usr/local/data/kvirji/offroad_navigation_dataset/depths/depth_{bag_name}_{file_number:04d}.npy'.format(bag_name=BAG_NAME, file_number=FILE_NUMBER), 'rb') as depthfile:
    depth_image = np.load(depthfile)

print(data)
cv2.imshow("DEPTH", depth_image)
cv2.imshow("RGB", rgb_image)
cv2.waitKey()
