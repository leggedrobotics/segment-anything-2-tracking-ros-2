"""
This scripts receive the image, process and publish the position of the rock
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import torch

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.subscription = self.create_subscription(
            Image,
            '/camera1/rgb',  # Replace with your camera topic
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Point, '/object_position', 10)
        self.bridge = CvBridge()

        # Tracking variables
        sam2_checkpoint = "/home/jonas/Coding/boulder_perception/segment-anything-2-real-time/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.if_init = False


    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width = frame.shape[:2]
        
        if not self.if_init:
            # Initialize SAM2 predictor with the first frame
            self.predictor.load_first_frame(frame)
            self.if_init = True

            # Define an initial bounding box (example coordinates, modify as needed)
            bbox = np.array([340, 550, 410, 500], dtype=np.float32)
            ann_frame_idx = 0
            ann_obj_id = 1
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
            )

            self.get_logger().info("Initialized SAM2 with bounding box")
        else:
            # Perform tracking
            out_obj_ids, out_mask_logits = self.predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255
                all_mask = cv2.bitwise_or(all_mask, out_mask)

            # Combine mask with the frame
            all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

            # Publish position of the tracked object (example center position)
            cx, cy = width // 2, height // 2  # Update with actual center computation if needed
            self.publish_position(cx, cy)

        # Visualize the result
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Tracked Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.destroy_node()

    def publish_position(self, x, y):
        # Publish object's position
        point = Point()
        point.x = 1.0
        point.y = 1.0
        point.z = 0.0
        self.publisher.publish(point)
        self.get_logger().info(f"Object position published: x={x}, y={y}")

def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    rclpy.spin(object_tracker)
    object_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
