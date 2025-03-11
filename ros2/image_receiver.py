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
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import torch
import yaml
import os
import time  # For timing and tracking updates

# Configure torch for optimal performance with CUDA devices.
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ObjectTracker(Node):
    """
    ROS2 node for object tracking using the SAM2 predictor.

    This node reads a configuration file (config.yaml) to set the camera and mask topics.
    It subscribes to the camera image topic to receive images and displays an OpenCV window
    for the user to click on a predefined number of points (default 5). Once the points are
    selected, it initializes the SAM2 predictor. The node then tracks objects in subsequent
    images and publishes a mask image along with a demo object position (the image center).

    The user may abort point selection by pressing 'q' in the selection window.
    """

    def __init__(self):
        super().__init__('object_tracker')

        # Load configuration from config.yaml located in the same folder as this script.
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Retrieve topic names from the configuration with defaults.
        camera_topic = config.get("camera_topic", "/camMainView/image_raw")
        mask_topic = config.get("mask_topic", "/src/mask")

        # Subscribe to the input camera topic (configured topic).
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        self.get_logger().info(f"Subscribed to camera topic: {camera_topic}")

        # Publisher for object position (for demonstration).
        self.publisher = self.create_publisher(Point, '/object_position', 10)
      
        # Publisher for mask images (configured topic).
        self.mask_publisher = self.create_publisher(Image, mask_topic, 10)
        self.get_logger().info(f"Publishing mask images on topic: {mask_topic}")

        # Bridge for converting between ROS Image messages and OpenCV images.
        self.bridge = CvBridge()

        # Initialize SAM2 predictor using pre-defined checkpoint and configuration.
        sam2_checkpoint = "/workspace/sam2_rt/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.if_init = False  # Flag to track whether SAM2 has been initialized.
        self.get_logger().info("SAM2 predictor built.")

        # Variables for user interaction:
        self.selected_points = []  # To store points selected by the user.
        self.frame = None          # Latest camera frame.
        self.wait_for_clicks = True  # Flag to wait for user input to select points.
        self.number_of_points = 5  # Number of points to be selected by the user.

        # Name of the window used for point selection.
        self.window_name = f"Select {self.number_of_points} Points"

    def image_callback(self, msg):
        """
        Callback for camera image topic.

        Converts the ROS Image message to an OpenCV image, waits for the user to select
        points if required, and either initializes the SAM2 predictor or performs object
        tracking on the incoming frame.
        """
        # Convert the ROS Image to an OpenCV image in BGR format and then to RGB.
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # If the node is waiting for user clicks to select points.
        if self.wait_for_clicks:
            self.get_logger().info(f"Selected points: {self.selected_points}")

            # Reset selected points each time a new frame is received.
            self.selected_points = []
            # Show the frame in a window for user interaction.
            cv2.imshow(self.window_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            self.get_logger().info(f"Please click on {self.number_of_points} points in the image...")
            # Wait until the required number of points is selected or user aborts.
            while len(self.selected_points) < self.number_of_points:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.get_logger().info("User aborted point selection.")
                    self.destroy_node()
                    return
            # Once points are selected, close the selection window.
            cv2.destroyWindow(self.window_name)
            self.get_logger().info(f"Selected points: {self.selected_points}")
            self.initialize_sam2(self.frame)
        else:
            # If SAM2 has been initialized, perform object tracking.
            self.track_objects(self.frame, msg)

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for point selection.

        When the left mouse button is clicked and the total selected points are less than the
        required number, records the (x, y) coordinate and provides visual feedback.
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < self.number_of_points:
            self.selected_points.append((x, y))
            self.get_logger().info(f"Point selected: ({x}, {y})")
            # Draw a small circle at the selected point for visualization.
            cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

    def initialize_sam2(self, frame):
        """
        Initialize the SAM2 predictor using the user-selected points.

        Converts the selected points to a numpy array, prepares the corresponding labels,
        loads the first frame into the predictor, and adds the prompt based on the points.
        """
        points = np.array(self.selected_points, dtype=np.float32)
        labels = np.ones(len(self.selected_points))  # Use ones as labels for foreground.
        ann_frame_idx = 0  # Index for the initial frame.
        ann_obj_id = 1     # Object ID to be assigned.

        # Load the first frame and initialize SAM2 with the new prompt.
        self.predictor.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )
        self.if_init = True         # Mark that SAM2 has been initialized.
        self.wait_for_clicks = False  # Stop waiting for further clicks.
        self.get_logger().info("Initialized SAM2 with user-selected points.")

    def track_objects(self, frame, img_msg):
        """
        Perform object tracking on the given frame using the SAM2 predictor.

        Processes the output masks from the predictor, merges them into a single mask,
        converts the mask for visualization, publishes the mask message, and publishes an
        example object position (center of the image).

        Args:
            frame (np.array): The current image frame in RGB format.
            img_msg (Image): The original ROS image message (used for header information).
        """
        start_time = time.time()  # Start timing the tracking process.
        # Use the predictor to track objects in the frame.
        out_obj_ids, out_mask_logits = self.predictor.track(frame)
        height, width = frame.shape[:2]

        # Create a blank grayscale mask.
        all_mask_gray = np.zeros((height, width), dtype=np.uint8)

        # Process and merge each object's mask.
        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 235
            all_mask_gray = cv2.bitwise_or(all_mask_gray, out_mask.squeeze())

        # Convert the grayscale mask to a BGR image for publishing.
        all_mask_bgr = cv2.cvtColor(all_mask_gray, cv2.COLOR_GRAY2BGR)
        mask_msg = self.bridge.cv2_to_imgmsg(all_mask_bgr, encoding='bgr8')
        # Copy header information from the input image.
        mask_msg.header.stamp = img_msg.header.stamp
        mask_msg.header.frame_id = img_msg.header.frame_id
        self.mask_publisher.publish(mask_msg)

        # For demonstration, overlay the mask on the original frame.
        all_mask_rgb = cv2.cvtColor(all_mask_gray, cv2.COLOR_GRAY2RGB)
        frame_overlay = cv2.addWeighted(frame, 1, all_mask_rgb, 0.5, 0)


        # Display the tracking result in an OpenCV window.
        cv2.imshow("Tracked Frame", cv2.cvtColor(frame_overlay, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("Tracking stopped by user.")
            self.destroy_node()

        duration_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"track_objects duration: {duration_ms:.2f} ms")


def main(args=None):
    """
    Main entry point for the object tracker node.

    Initializes the ROS node, creates an ObjectTracker instance, and enters the spin loop.
    """
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
