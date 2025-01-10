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

        # SAM2 model initialization
        sam2_checkpoint = "/home/jonas/Coding/boulder_perception/segment-anything-2-real-time/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"

        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.if_init = False

        # User interaction variables
        self.selected_points = []
        self.frame = None
        self.wait_for_clicks = True  # Wait for user input
        self.number_of_points = 5

        # Display window for user clicks
        self.window_name = f"Select {self.number_of_points} Points"

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

        # Wait for user to select 2 points
        if self.wait_for_clicks:
            self.selected_points = []
            cv2.imshow(self.window_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            cv2.setMouseCallback(self.window_name, self.mouse_callback)

            self.get_logger().info(f"Please click on {self.number_of_points} points in the image...")
            while len(self.selected_points) < self.number_of_points:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.get_logger().info("User aborted point selection.")
                    self.destroy_node()
                    return
            cv2.destroyWindow(self.window_name)
            self.get_logger().info(f"Selected points: {self.selected_points}")
            self.initialize_sam2(self.frame)

        else:
            # Perform tracking
            self.track_objects(self.frame)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < self.number_of_points:
            print(len(self.selected_points), self.number_of_points)
            self.selected_points.append((x, y))
            self.get_logger().info(f"Point selected: ({x}, {y})")
            # Visualize the selected point
            cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(self.window_name, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

    def initialize_sam2(self, frame):
        points = np.array(self.selected_points, dtype=np.float32)
        labels = np.ones(len(self.selected_points))  # Labels: foreground
        ann_frame_idx = 0
        ann_obj_id = 1

        self.predictor.load_first_frame(frame)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )
        self.if_init = True
        self.wait_for_clicks = False
        self.get_logger().info("Initialized SAM2 with user-selected points.")

    def track_objects(self, frame):
        out_obj_ids, out_mask_logits = self.predictor.track(frame)
        height, width = frame.shape[:2]
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)

        for i in range(len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)

        # Combine mask with the frame
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        # Publish example position
        cx, cy = width // 2, height // 2
        self.publish_position(cx, cy)

        # Show tracking result
        cv2.imshow("Tracked Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.get_logger().info("Tracking stopped by user.")
            self.destroy_node()

    def publish_position(self, x, y):
        point = Point()
        point.x = float(x)
        point.y = float(y)
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
