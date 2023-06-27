#!/usr/bin/env python3
import rospy
import rospkg
import os
import yaml
import message_filters
from sensor_msgs.msg import Image, NavSatFix, Imu
from rccar_controller_pc.msg import TeensySerial
import cv2
from cv_bridge import CvBridge
import numpy as np


class RCDataSubscriber:

    def __init__(self, max_delay, bag_name):
        # synchronized topics
        self.image_subscriber = message_filters.Subscriber('/d400/color/image_raw', Image)
        self.depth_subscriber = message_filters.Subscriber('/d400/aligned_depth_to_color/image_raw', Image)
        self.teensy_subscriber = message_filters.Subscriber('/teensy_serial', TeensySerial)
        self.imu_subscriber = message_filters.Subscriber('/mavros/imu/data', Imu)
        self.gps_subscriber = message_filters.Subscriber('/gps', NavSatFix)
        self.bridge = CvBridge()

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_subscriber, self.teensy_subscriber, self.imu_subscriber, self.gps_subscriber,
             self.depth_subscriber], 25, max_delay)
        self.ts.registerCallback(self.rc_callback)

        # unsynchronized topics
        self.image_subscriber_non_sync = rospy.Subscriber('/d400/color/image_raw', Image, self.image_callback)
        self.depth_subscriber_non_sync = rospy.Subscriber('/d400/color/image_raw', Image, self.depth_callback)
        self.teensy_subscriber = rospy.Subscriber('/teensy_serial', TeensySerial, self.teensy_callback)
        self.imu_subscriber_non_sync = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback)
        self.gps_subscriber = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)

        self.file_counter = 0
        self.save = True
        self.bag_name = bag_name

        # For verifying synchronization
        self.gps_counter = 0
        self.sync_counter = 0

        self.teensy_data = None
        self.imu_data = None
        self.image_data = None
        self.depth_data = None

        self.average_delay_sync = 0
        self.n_sync = 0

        self.average_delay_non_sync = 0
        self.n = 0

    def rc_callback(self, image, teensy, imu, gps, depth):
        if image is None or teensy is None or imu is None or gps is None or depth is None:
            return

        # display
        text = "TEENSY: ({steering_angle:.2f},{throttle:.2f}). IMU: ({w:.2f},{x:.2f},{y:.2f},{z:.2f}), GPS: ({lat:.3f},{lon:.3f})" \
            .format(steering_angle=teensy.ch_str, throttle=teensy.ch_thr, w=imu.orientation.w, x=imu.orientation.x,
                    y=imu.orientation.y,
                    z=imu.orientation.z, lat=gps.latitude, lon=gps.longitude)
        self.display(image, text)

        # analyze time delay
        r = self.max_diff([image, teensy, imu, gps, depth])
        if self.n_sync == 0:
            self.average_delay_sync = r
        else:
            self.average_delay_sync += (1 / self.n_sync) * (r - self.average_delay_sync)
        self.n_sync += 1
        print("AVERAGE TIME DELAY (s): ", self.average_delay_sync)

        # verify count
        self.sync_counter += 1
        print("SYNC COUNT: ", self.sync_counter)

        # save synchronized data to dataset
        if self.save:
            self.file_counter += 1
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth)
            cv_rgb_image = self.bridge.imgmsg_to_cv2(image)
            cv_rgb_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_BGR2RGB)

            data = np.asarray([teensy.ch_str, teensy.ch_thr, imu.orientation.w, imu.orientation.x, imu.orientation.y,
                               imu.orientation.z,
                               gps.longitude, gps.latitude, image.header.stamp.to_sec(), depth.header.stamp.to_sec(),
                               teensy.header.stamp.to_sec(),
                               imu.header.stamp.to_sec(), gps.header.stamp.to_sec()])

            with open('topic_data/topics_{bag_name}_{file_counter:04d}.npy'.format(bag_name=self.bag_name, file_counter=self.file_counter), 'wb') as datafile:
                np.save(datafile, data)
            with open('depth_data/depth_{bag_name}_{file_counter:04d}.npy'.format(bag_name=self.bag_name, file_counter=self.file_counter), 'wb') as depthfile:
                np.save(depthfile, cv_depth_image)
            cv2.imwrite('image_data/image_{bag_name}_{file_counter:04d}.jpg'.format(bag_name=self.bag_name, file_counter=self.file_counter), cv_rgb_image)
        return

    # display image with text
    def display(self, image, text=None):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv2.putText(cv_image, text, (10, 460), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Dataset", cv_image)
        cv2.waitKey(1)
        return

    # returns max time delay between a list of topics
    def max_diff(self, data):
        data.sort(key=lambda d: d.header.stamp)
        return abs(data[-1].header.stamp - data[0].header.stamp).to_sec()

    def image_callback(self, image):
        self.image_data = image
        return

    def depth_callback(self, depth):
        self.depth_data = depth

    def teensy_callback(self, teensy):
        self.teensy_data = teensy
        return

    def imu_callback(self, imu):
        self.imu_data = imu
        return

    def gps_callback(self, gps):
        if self.image_data is None or self.imu_data is None or self.teensy_data is None or self.depth_data is None:
            return
        r = self.max_diff([self.image_data, self.teensy_data, self.imu_data, gps, self.depth_data])
        if self.n == 0:
            self.average_delay_non_sync = r
        else:
            self.average_delay_non_sync += (1 / self.n) * (r - self.average_delay_non_sync)
        self.n += 1
        print("AVERAGE TIME DELAY WITHOUT SYNC: ", self.average_delay_non_sync)
        self.gps_counter += 1
        print("NON SYNC COUNT: ", self.gps_counter)
        return


if __name__ == '__main__':
    rospy.init_node('rc_subscriber')
    rospy.loginfo("rc_subscriber node initialization")

    with open(os.path.join(rospkg.RosPack().get_path('extract_dataset'), 'config/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    rc_data_subscriber = RCDataSubscriber(config['max_time_delay_between_topics'], config['bag_name'])

    rospy.spin()
