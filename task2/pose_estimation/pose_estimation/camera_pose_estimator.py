import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class CameraPoseEstimator(Node):
    def __init__(self):
        super().__init__('camera_pose_estimator_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Hardcoded Camera Calibration for Simulated Camera
        # Assume 640x480 image, focal lengths ~554.3827, center at (320.5, 240.5)
        self.K = np.array([
            [554.3827, 0.0, 320.5],
            [0.0, 554.3827, 240.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # No distortion (ideal camera in simulation)
        self.D = np.zeros(5)

        # Hardcoded 2D Image Points (pixel coordinates clicked manually)
        self.image_points = np.array([
            [320, 240],  # bottom-front-left
            [400, 240],  # bottom-front-right
            [400, 300],  # top-back-right
            [320, 300],  # top-back-left
        ], dtype=np.float32)

        # Corresponding 3D Object Points in World (cube corners in meters)
        self.object_points = np.array([
            [-0.68, 0.50, 0.95],  # bottom-front-left
            [-0.68, 0.80, 0.95],  # bottom-front-right
            [-0.68, 0.80, 1.01],  # top-back-right
            [-0.68, 0.50, 1.01],  # top-back-left
        ], dtype=np.float32)

        # Subscribe to the live camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.process_camera_image,
            10
        )

    def process_camera_image(self, msg):
        # Convert incoming ROS image message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Solve PnP (Perspective-n-Point) problem
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            self.image_points,
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            self.get_logger().info('✅ Camera pose estimation successful.')

            # Log the estimated object pose (rotation and translation)
            self.get_logger().info(f'Rotation Vector (rvec):\n{rvec.flatten()}')
            self.get_logger().info(f'Translation Vector (tvec):\n{tvec.flatten()}')

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            self.get_logger().info(f'Rotation Matrix:\n{R}')

            # Optional: Draw coordinate frame on the image
            cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)

        else:
            self.get_logger().warn('⚠️ Pose estimation failed this frame.')

        # Show the annotated camera frame
        cv2.imshow('TurtleBot3 Camera View', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseEstimator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

