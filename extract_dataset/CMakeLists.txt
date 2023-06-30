cmake_minimum_required(VERSION 2.8.3)
project(extract_dataset)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  message_filters
)


catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs message_filters
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
   rc_subscriber_node.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
        )

