#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np

class PointSelector(Node):
    def __init__(self):
        super().__init__('point_selector')
        # Subscribe to the raw camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/camMainView/image_raw',  # Change if needed
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
        self.selection_done = False

        # Create the window and set the mouse callback once
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Use a timer callback to update the display and check for completion
        self.timer = self.create_timer(0.1, self.timer_callback)

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
        # Show the image in a window (convert back to BGR for OpenCV display)
        cv2.imshow(self.window_name, cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("User aborted point selection.")
            self.destroy_node()

        # When enough points have been selected, publish them
        if len(self.selected_points) >= self.number_of_points and not self.selection_done:
            self.get_logger().info(f"Selected points: {self.selected_points}")
            poly = Polygon()
            for pt in self.selected_points:
                p = Point32()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = 0.0
                poly.points.append(p)
            self.publisher.publish(poly)
            self.get_logger().info("Published selected points on '/selected_points'.")
            self.selection_done = True
            cv2.destroyWindow(self.window_name)

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
