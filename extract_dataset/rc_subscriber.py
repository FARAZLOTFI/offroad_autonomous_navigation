#!/usr/bin/env python3
import rospy
import rospkg
import os
import yaml
import message_filters
from sensor_msgs.msg import Image
from rccar_controller_pc.msg import TeensySerial


class RCDataSubscriber:

    def __init__(self, max_delay):
        # synchronized topics
        self.image_subscriber = message_filters.Subscriber('/d400/color/image_raw', Image)
        self.teensy_subscriber = message_filters.Subscriber('/teensy_serial', TeensySerial)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber, self.teensy_subscriber], 10, max_delay)
        self.ts.registerCallback(self.rc_callback)

        # unsynchronized topics
        # self.image_subscriber_non_sync = rospy.Subscriber('/d400/color/image_raw', Image, self.image_callback)
        # self.teensy_subscriber = rospy.Subscriber('/teensy_serial', TeensySerial, self.teensy_callback)

        # For verifying synchronization
        # self.sync_counter = 0
        # self.image_counter = 0
        # self.teensy_counter = 0
        # self.image_stamp = 0
        # self.sync_diff = 0
    def rc_callback(self, image, data):
        # save synchronized data
        return

    # def image_callback(self, image):
    #     return
    #
    # def teensy_callback(self, data):
    #     return


if __name__ == '__main__':
    rospy.init_node('rc_subscriber')
    rospy.loginfo("rc_subscriber node initialization")

    with open(os.path.join(rospkg.RosPack().get_path('extract_dataset'), 'config/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    r = rospy.Rate(config['rate'])
    rc_data_subscriber = RCDataSubscriber(config['max_time_delay_between_topics'])
    while not rospy.is_shutdown():
        r.sleep()
