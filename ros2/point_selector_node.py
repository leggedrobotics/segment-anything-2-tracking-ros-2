#!/usr/bin/env python3
# MIT License
# 
# Copyright (c) 2025 Jonas Grütter
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# (Full MIT License text should be included in the LICENSE file)
#
# This file is part of an open-source object tracking node using SAM2.
# The node subscribes to a camera image topic, loads a configuration file,
# and allows the user to manually select points from the image. These points
# are then used to initialize a SAM2 predictor for object tracking. The node
# publishes both a mask image and an example object position (center of the image).
#
# For more details and instructions, please refer to the project README.
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
class PointSelector(Node):
    def __init__(self):
        super().__init__('point_selector')
        # Load configuration from config.yaml located in the same folder as this script.
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Retrieve topic names from the configuration with defaults.
        camera_topic = config.get("camera_topic", "/camMainView/image_raw")
        
        # Subscribe to the raw camera image topic
        self.subscription = self.create_subscription(
            Image,
            camera_topic,  # Change if needed
            self.image_callback,
            10
        )
        # Publisher to send the selected points
        self.publisher = self.create_publisher(Polygon, '/selected_points', 10)
        self.bridge = CvBridge()
        self.latest_frame = None

        # Variables for user interaction
        self.selected_points = []
        self.number_of_points = 5  # Adjust as needed
        self.window_name = f"Select {self.number_of_points} Points"

        # Create the window and set the mouse callback once
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Timer for updating the display (e.g., 10 Hz)
        self.display_timer = self.create_timer(0.1, self.timer_callback)
        # Timer for publishing points at a lower frequency (e.g., 1 Hz)
        self.publish_timer = self.create_timer(1.0, self.publish_points_callback)

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB for consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.latest_frame = frame_rgb
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def timer_callback(self):
        if self.latest_frame is None:
            return

        # Create a copy to draw the selected points
        disp_frame = self.latest_frame.copy()
        for pt in self.selected_points:
            cv2.circle(disp_frame, (pt[0], pt[1]), 5, (255, 0, 0), -1)
        # Display the image (convert back to BGR for OpenCV display)
        cv2.imshow(self.window_name, cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("User aborted point selection.")
            self.destroy_node()

    def publish_points_callback(self):
        # Publish only when the required number of points has been selected
        if len(self.selected_points) >= self.number_of_points:
            poly = Polygon()
            for pt in self.selected_points:
                p = Point32()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = 0.0
                poly.points.append(p)
            self.publisher.publish(poly)
            self.get_logger().info("Published selected points on '/selected_points'.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < self.number_of_points:
            self.selected_points.append((x, y))
            self.get_logger().info(f"Point selected: ({x}, {y})")

def main(args=None):
    rclpy.init(args=args)
    point_selector = PointSelector()
    rclpy.spin(point_selector)
    point_selector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
