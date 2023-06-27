import numpy as np
import cv2

FILE_NUMBER = 260
BAG_NAME = '2023-06-13-09-56-16'

with open('/home/farnoosh/rc_car/topic_data/topics_{bag_name}_{file_number:04d}.npy'.format(bag_name= BAG_NAME, file_number=FILE_NUMBER), 'rb') as datafile:
    data = np.load(datafile)

rgb_image = cv2.imread('/home/farnoosh/rc_car/image_data/image_{bag_name}_{file_number:04d}.jpg'.format(bag_name= BAG_NAME, file_number=FILE_NUMBER))

with open('/home/farnoosh/rc_car/depth_data/depth_{bag_name}_{file_number:04d}.npy'.format(bag_name=BAG_NAME, file_number=FILE_NUMBER), 'rb') as depthfile:
    depth_image = np.load(depthfile)

print(data)
cv2.imshow("DEPTH", depth_image)
cv2.imshow("RGB", rgb_image)
cv2.waitKey()