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

# The node subscribes to a camera image topic and a "/selected_points" topic.
# It uses the selected points to initialize a SAM2 predictor for object tracking.
# If new selected points are not received for more than 5 seconds (i.e. five missed updates),
# the predictor is reset and the node waits for a new set of selected points to reinitialize.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor
import time  # For timing and tracking updates

# Setup torch settings (adjust as needed for your CUDA device)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ObjectTracker(Node):
    """
    ROS2 node for object tracking using the SAM2 predictor.

    This node subscribes to two topics:
      - '/camMainView/image_raw': receives camera images.
      - '/selected_points': receives a set of points (as a Polygon) that define the object(s) to track.
    
    On receiving new images, the node either:
      1. Initializes the SAM2 predictor (if not already initialized) using the latest selected points.
      2. Tracks objects using the predictor if it has been initialized.

    If no new selected points are received for more than 5 seconds, the predictor resets its state,
    and the node waits for new selected points to reinitialize.
    """

    def __init__(self):
        super().__init__('object_tracker')

        # Subscribe to the camera image topic.
        self.subscription = self.create_subscription(
            Image,
            '/camMainView/image_raw',  # Topic name for camera images.
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to topic: /camMainView/image_raw")

        # Subscribe to the selected points topic.
        self.points_sub = self.create_subscription(
            Polygon,
            '/selected_points',
            self.points_callback,
            10
        )
        self.get_logger().info("Subscribed to topic: /selected_points")

        # Publisher for output mask images.
        self.mask_publisher = self.create_publisher(Image, '/src/mask', 10)
        self.get_logger().info("Publishing to topic: /src/mask")

        # Bridge to convert between ROS Image messages and OpenCV images.
        self.bridge = CvBridge()

        # Store selected points (as a NumPy array) once received.
        self.selected_points = None
        # Flag indicating whether SAM2 has been initialized.
        self.if_init = False
        # Timestamp when the last selected points were received.
        self.last_points_time = None

        # Initialize the SAM2 predictor.
        # Adjust the checkpoint path and model configuration file as needed.
        sam2_checkpoint = "/workspace/sam2_rt/checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.get_logger().info("SAM2 predictor built.")

    def points_callback(self, msg):
        """
        Callback for the '/selected_points' topic.

        Converts the incoming Polygon message to a numpy array of (x,y) coordinates and updates the
        selected points used for initialization of the SAM2 predictor. It also updates the timestamp
        for when the points were last received.
        """
        self.get_logger().debug("points_callback triggered")
        pts = []
        for p in msg.points:
            pts.append((int(p.x), int(p.y)))
        self.selected_points = np.array(pts, dtype=np.float32)
        # Record the current time as the latest update time.
        self.last_points_time = time.time()
        self.get_logger().info(f"Received selected points: {self.selected_points}")

    def image_callback(self, msg):
        """
        Callback for the camera image topic.

        Converts the ROS Image message to an OpenCV image (RGB format), and based on the state of the SAM2 predictor:
          - If the predictor is not initialized and selected points are available, initialize SAM2.
          - If the predictor is initialized, check if new selected points have been received within the last 5 seconds.
            If not, reset the predictor.
          - Otherwise, use the predictor to track objects in the new image frame.
        """
        self.get_logger().debug("image_callback triggered")
        try:
            # Convert ROS Image message to OpenCV image.
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.get_logger().debug("Image converted to RGB")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Check if the predictor is already initialized.
        if self.if_init and self.last_points_time is not None:
            # If more than 5 seconds have elapsed without new selected points, reset the predictor.
            if time.time() - self.last_points_time > 5.0:
                self.get_logger().info("No new selected points received for 5 seconds. Resetting SAM2 predictor.")
                self.predictor.reset_state()
                self.if_init = False
                self.selected_points = None

        # If the predictor is not initialized and selected points are available, initialize SAM2.
        if not self.if_init:
            if self.selected_points is not None:
                self.get_logger().info("Initializing SAM2 with current frame and selected points")
                self.initialize_sam2(frame_rgb)
            else:
                self.get_logger().debug("Selected points not yet received. Skipping initialization.")
        else:
            # If initialized, perform object tracking.
            self.get_logger().debug("Tracking objects on new frame")
            self.track_objects(frame_rgb, msg)

    def initialize_sam2(self, frame):
        """
        Initialize the SAM2 predictor using the current frame and the selected points.

        This function prepares a label array (all ones, assuming a single foreground class),
        calls load_first_frame on the predictor, and adds the new prompt using the selected points.
        """
        # Create an array of labels (all ones) corresponding to each selected point.
        labels = np.ones(len(self.selected_points))
        ann_frame_idx = 0  # Frame index for initialization.
        ann_obj_id = 1     # Object id for initialization.
        self.predictor.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id,
            points=self.selected_points, labels=labels
        )
        # Mark the predictor as initialized.
        self.if_init = True
        self.get_logger().info("Initialized SAM2 with selected points.")

    def track_objects(self, frame, img_msg):
        """
        Perform object tracking using the SAM2 predictor on the given frame.

        This function calls the predictor's track() method, processes the output masks,
        creates a combined mask image, and publishes it on the '/src/mask' topic.

        Args:
            frame (np.array): The current frame in RGB format.
            img_msg (Image): The original ROS image message (used for copying header information).
        """
        start_time = time.time()  # Start timing the tracking process.
        out_obj_ids, out_mask_logits = self.predictor.track(frame)
        self.get_logger().debug(f"Track output: {len(out_obj_ids)} objects detected")
        height, width = frame.shape[:2]
        # Create an empty grayscale mask.
        all_mask_gray = np.zeros((height, width), dtype=np.uint8)
        # Process each object's mask.
        for i in range(len(out_obj_ids)):
            try:
                # Convert the tensor mask to a numpy array.
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy()
                out_mask = out_mask.astype(np.uint8) * 235  # Scale mask intensity.
                # Merge the mask using bitwise OR.
                all_mask_gray = cv2.bitwise_or(all_mask_gray, out_mask.squeeze())
                self.get_logger().debug(f"Processed mask for object {i}: shape {out_mask.shape}")
            except Exception as e:
                self.get_logger().error(f"Error processing mask for object {i}: {e}")
                continue

        # Convert the grayscale mask to a BGR image for visualization.
        try:
            all_mask_bgr = cv2.cvtColor(all_mask_gray, cv2.COLOR_GRAY2BGR)
            mask_msg = self.bridge.cv2_to_imgmsg(all_mask_bgr, encoding='bgr8')
            # Copy header information (timestamp, frame id) from the original image.
            mask_msg.header.stamp = img_msg.header.stamp
            mask_msg.header.frame_id = img_msg.header.frame_id
            # Publish the mask image.
            self.mask_publisher.publish(mask_msg)
            self.get_logger().info("Published mask message on /src/mask")
        except Exception as e:
            self.get_logger().error(f"Error publishing mask image: {e}")

        duration_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"track_objects duration: {duration_ms:.2f} ms")

def main(args=None):
    """
    Main entry point for the object tracker node.

    Initializes the ROS node, creates an ObjectTracker instance, and spins the node.
    """
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
