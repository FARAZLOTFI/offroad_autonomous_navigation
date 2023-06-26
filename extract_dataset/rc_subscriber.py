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

class RCDataSubscriber:

    def __init__(self, max_delay):
        # synchronized topics
        self.image_subscriber = message_filters.Subscriber('/d400/color/image_raw', Image)
        self.teensy_subscriber = message_filters.Subscriber('/teensy_serial', TeensySerial)
        self.imu_subscriber = message_filters.Subscriber('/mavros/imu/data', Imu)
        self.gps_subscriber = message_filters.Subscriber('/gps', NavSatFix)
        self.bridge = CvBridge()

        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber, self.teensy_subscriber, self.imu_subscriber, self.gps_subscriber], 25, max_delay)
        self.ts.registerCallback(self.rc_callback)

        # unsynchronized topics
        # self.image_subscriber_non_sync = rospy.Subscriber('/d400/color/image_raw', Image, self.image_callback)
        # self.teensy_subscriber = rospy.Subscriber('/teensy_serial', TeensySerial, self.teensy_callback)
        # self.imu_subscriber_non_sync = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback)
        # self.gps_subscriber = rospy.Subscriber('/gps', NavSatFix, self.gps_callback)

        # For verifying synchronization
        # self.gps_counter = 0
        # self.sync_counter = 0
        #
        # self.teensy_data = None
        # self.imu_data = None
        # self.image_data = None
        #
        self.average_delay_sync = 0
        self.n_sync = 0
        #
        # self.average_delay_non_sync = 0
        # self.n = 0

    def rc_callback(self, image, teensy, imu, gps):
        # get synchronized data
        orientation = {"w": imu.orientation.w, "x": imu.orientation.x, "y": imu.orientation.y, "z": imu.orientation.z}
        gps_coordinates = {"lat": gps.latitude, "lon": gps.longitude}
        controls = {"steering_angle": teensy.ch_str, "throttle": teensy.ch_thr}

        # display
        text = "TEENSY: ({steering_angle:.2f},{throttle:.2f}). IMU: ({w:.2f},{x:.2f},{y:.2f},{z:.2f}), GPS: ({lat:.3f},{lon:.3f})" \
            .format(steering_angle=controls["steering_angle"], throttle=controls["throttle"], w=orientation["w"], x=orientation["x"],
                    y=orientation["y"],
                    z=orientation["z"], lat=gps_coordinates["lat"], lon=gps_coordinates["lon"])
        self.display(image, text)

        # analyze time delay
        r = self.max_diff([image, teensy, imu, gps])
        if self.n_sync == 0:
            self.average_delay_sync = r
        else:
            self.average_delay_sync += (1/self.n_sync) * (r - self.average_delay_sync)
        self.n_sync += 1
        print("AVERAGE TIME DELAY (s): ",  self.average_delay_sync, end='\r')

        # verify count
        # self.sync_counter += 1
        # print("GPS_COUNT SYNC: ", self.sync_counter)
        return

    # display image with text
    def display(self, image, text=None):
        cv_image = self.bridge.imgmsg_to_cv2(image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv2.putText(cv_image, text, (10,460), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Dataset", cv_image)
        cv2.waitKey(1)
        return

    # returns max time delay between a list of topics
    def max_diff(self, data):
        data.sort(key=lambda d: d.header.stamp)
        return abs(data[-1].header.stamp - data[0].header.stamp).to_sec()

    # def image_callback(self, image):
    #     self.image_data = image
    #     return
    #
    # def teensy_callback(self, teensy):
    #     self.teensy_data = teensy
    #     return
    #
    # def imu_callback(self, imu):
    #     self.imu_data = imu
    #     return
    #
    # def gps_callback(self, gps):
    #     if self.image_data is None or self.imu_data is None or self.teensy_data is None:
    #         return
    #     r = self.max_diff([self.image_data, self.teensy_data, self.imu_data, gps])
    #     if self.n == 0:
    #         self.average_delay_non_sync = r
    #     else:
    #         self.average_delay_non_sync += (1 / self.n) * (r - self.average_delay_non_sync)
    #     self.n += 1
    #     print("NON SYNC: :", self.average_delay_non_sync)
    #     self.gps_counter += 1
    #     print("GPS_COUNT NON SYNC: ", self.gps_counter)
    #     return


if __name__ == '__main__':
    rospy.init_node('rc_subscriber')
    rospy.loginfo("rc_subscriber node initialization")

    with open(os.path.join(rospkg.RosPack().get_path('extract_dataset'), 'config/config.yaml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    rc_data_subscriber = RCDataSubscriber(config['max_time_delay_between_topics'])
    rospy.spin()
