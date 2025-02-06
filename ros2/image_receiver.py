import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import torch

# Setup torch settings
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        self.subscription = self.create_subscription(
            Image,
            '/camMainView/image_raw',  # Replace with your camera topic
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Point, '/object_position', 10)
        
        # **Mask publisher** on topic '/src/mask'
        self.mask_publisher = self.create_publisher(Image, '/src/mask', 10)

        self.bridge = CvBridge()

        # SAM2 model initialization
        sam2_checkpoint = "/workspace/sam2_rt/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.if_init = False

        # User interaction variables
        self.selected_points = []
        self.frame = None
        self.wait_for_clicks = True
        self.number_of_points = 5

        # Create window for point selection and set mouse callback
        self.window_name = f"Select {self.number_of_points} Points"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Timer to update display and perform tracking (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Track last image reception time
        self.last_image_time = self.get_clock().now()

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame = frame
            self.last_image_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def timer_callback(self):
        # Warn if no image received in the last 5 seconds
        now = self.get_clock().now()
        dt = (now - self.last_image_time).nanoseconds / 1e9
        if dt > 5.0:
            self.get_logger().warn("No image received in the last 5 seconds.")

        if self.frame is None:
            return

<<<<<<< HEAD
=======
        # Wait for user to select points
>>>>>>> 09f1a42 (changed to publish mask)
        if self.wait_for_clicks:
            # Display the frame with any selected points
            display_frame = self.frame.copy()
            for pt in self.selected_points:
                cv2.circle(display_frame, pt, 5, (255, 0, 0), -1)
            cv2.imshow(self.window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("User aborted point selection.")
                self.destroy_node()
                rclpy.shutdown()
                return
            # Once enough points are selected, initialize SAM2
            if len(self.selected_points) >= self.number_of_points:
                cv2.destroyWindow(self.window_name)
                self.initialize_sam2(self.frame)
        else:
            # Tracking mode
            if self.if_init:
                self.track_objects(self.frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Tracking stopped by user.")
                    self.destroy_node()
                    rclpy.shutdown()

    def mouse_callback(self, event, x, y, flags, param):
        if self.wait_for_clicks and event == cv2.EVENT_LBUTTONDOWN and len(self.selected_points) < self.number_of_points:
            self.selected_points.append((x, y))
            self.get_logger().info(f"Point selected: ({x}, {y})")

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
        all_mask = np.zeros((height, width), dtype=np.uint8)

        # Prepare a blank single-channel mask
        all_mask_gray = np.zeros((height, width), dtype=np.uint8)

        # Merge all object masks
        for i in range(len(out_obj_ids)):
            mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 235
            all_mask = cv2.bitwise_or(all_mask, mask)

        colored_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        combined_frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Publish a sample position (center of the image)
        cx, cy = width // 2, height // 2
        self.publish_position(cx, cy)

        cv2.imshow("Tracked Frame", cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

    def publish_position(self, x, y):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
        self.publisher.publish(point)
        self.get_logger().info(f"Published object position: x={x}, y={y}")


def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()
    try:
        rclpy.spin(object_tracker)
    except KeyboardInterrupt:
        object_tracker.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        object_tracker.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
