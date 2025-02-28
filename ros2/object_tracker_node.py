#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor
import time  # For timing the function execution

# Setup torch settings (adjust as needed for your CUDA device)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        # Subscribe to the camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/camMainView/image_raw',  # Change if needed
            self.image_callback,
            10
        )
        self.get_logger().info("Subscribed to topic: /camMainView/image_raw")

        # Subscribe to the selected points published by the point selector node
        self.points_sub = self.create_subscription(
            Polygon,
            '/selected_points',
            self.points_callback,
            10
        )
        self.get_logger().info("Subscribed to topic: /selected_points")

        # Removed publisher for object position
        self.mask_publisher = self.create_publisher(Image, '/src/mask', 10)
        self.get_logger().info("Publishing to topic: /src/mask")

        self.bridge = CvBridge()

        # To store the selected points once received
        self.selected_points = None
        # Flag to mark if SAM2 has been initialized
        self.if_init = False

        # Initialize the SAM2 predictor (adjust paths as needed)
        sam2_checkpoint = "/workspace/sam2_rt/checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.get_logger().info("SAM2 predictor built.")

    def points_callback(self, msg):
        self.get_logger().debug("points_callback triggered")
        # Convert the Polygon message to a numpy array of (x,y) coordinates
        pts = []
        for p in msg.points:
            pts.append((int(p.x), int(p.y)))
        self.selected_points = np.array(pts, dtype=np.float32)
        self.get_logger().info(f"Received selected points: {self.selected_points}")

    def image_callback(self, msg):
        self.get_logger().debug("image_callback triggered")
        try:
            # Convert the incoming image message
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.get_logger().debug("Image converted to RGB")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # If SAM2 is not yet initialized and we have received the selected points, initialize
        if not self.if_init:
            if self.selected_points is not None:
                self.get_logger().info("Initializing SAM2 with current frame and selected points")
                self.initialize_sam2(frame_rgb)
            else:
                self.get_logger().debug("Selected points not yet received. Skipping initialization.")
        else:
            # Otherwise, perform tracking and pass the original msg header along
            self.get_logger().debug("Tracking objects on new frame")
            self.track_objects(frame_rgb, msg)

    def initialize_sam2(self, frame):
        # Prepare a label array (here all ones for foreground)
        labels = np.ones(len(self.selected_points))
        ann_frame_idx = 0
        ann_obj_id = 1
        self.predictor.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id,
            points=self.selected_points, labels=labels
        )
        self.if_init = True
        self.get_logger().info("Initialized SAM2 with selected points.")

    def track_objects(self, frame, img_msg):
        start_time = time.time()  # Start timing the tracking process
        out_obj_ids, out_mask_logits = self.predictor.track(frame)
        self.get_logger().debug(f"Track output: {len(out_obj_ids)} objects detected")
        height, width = frame.shape[:2]
        # Create an empty mask
        all_mask_gray = np.zeros((height, width), dtype=np.uint8)
        # Merge all object masks
        for i in range(len(out_obj_ids)):
            try:
                # Convert the tensor mask to numpy array; expected tensor shape is [1, H, W]
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy()
                out_mask = out_mask.astype(np.uint8) * 235
                all_mask_gray = cv2.bitwise_or(all_mask_gray, out_mask.squeeze())
                self.get_logger().debug(f"Processed mask for object {i}: shape {out_mask.shape}")
            except Exception as e:
                self.get_logger().error(f"Error processing mask for object {i}: {e}")
                continue

        # Prepare a color mask and publish it with the same header as the original image message
        try:
            all_mask_bgr = cv2.cvtColor(all_mask_gray, cv2.COLOR_GRAY2BGR)
            mask_msg = self.bridge.cv2_to_imgmsg(all_mask_bgr, encoding='bgr8')
            # Copy the header (stamp and frame_id) from the original image message
            mask_msg.header.stamp = img_msg.header.stamp
            mask_msg.header.frame_id = img_msg.header.frame_id
            self.mask_publisher.publish(mask_msg)
            self.get_logger().info("Published mask message on /src/mask")
        except Exception as e:
            self.get_logger().error(f"Error publishing mask image: {e}")

        # Compute and log the duration of the tracking operation in milliseconds
        duration_ms = (time.time() - start_time) * 1000
        self.get_logger().info(f"track_objects duration: {duration_ms:.2f} ms")

def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
