import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import math
from scipy.spatial.transform import Rotation

class PosePublisher(Node):
    def __init__(self):
        """
        Initialize the PosePublisher node.

        Sets up the ROS2 node, creates a publisher for pose messages,
        and initializes a timer for periodic publishing.

        Args:
            None

        Returns:
            None
        """
        super().__init__('pose_publisher')

        # Create a publisher on the '/initialpose' topic
        self.publisher_ = self.create_publisher(PoseStamped, '/planning/mission_planning/goal', 10)

        # Create a timer to call the publish_pose method at a specified interval
        self.timer = self.create_timer(1.0, self.publish_pose)  # Publish every 1 second

    def publish_pose(self):
        """
        Publish a pose message to the '/planning/mission_planning/goal' topic.

        Creates and publishes a PoseStamped message using the stored pose data.
        Logs the published message for debugging purposes.

        Args:
            None

        Returns:
            None
        """
        # Create the message
        pose_msg = PoseStamped()

        # Populate the header
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = self.pose_msg['x']
        pose_msg.pose.position.y = self.pose_msg['y']
        pose_msg.pose.position.z = self.pose_msg['z']

        pose_msg.pose.orientation.x = self.pose_msg['x1']
        pose_msg.pose.orientation.y = self.pose_msg['y1']
        pose_msg.pose.orientation.z = self.pose_msg['z1']
        pose_msg.pose.orientation.w = self.pose_msg['w']

        # Publish the message
        self.publisher_.publish(pose_msg)
        self.get_logger().info('Publishing: {}'.format(pose_msg))
        
    def convert_from_carla_to_autoware(self, carla_point, autoware_point=None):
        """
        Convert CARLA coordinate system pose to Autoware coordinate system.

        Takes a CARLA Transform object and converts it to Autoware's coordinate
        system, storing the result for later publishing.

        Args:
            carla_point (carla.Transform): The CARLA pose to convert.
            autoware_point (dict, optional): Pre-converted Autoware pose data.
                If provided, this will be used instead of converting carla_point.

        Returns:
            None: The converted pose is stored in self.pose_msg.
        """
        if autoware_point:
            print("target point chosen from autowarePoint \n")
            self.pose_msg = autoware_point
            return
        carla_location = carla_point.location 
        carla_rotation = carla_point.rotation
        # Convert CARLA coordinates to Autoware coordinates
        autoware_x = carla_location.x
        autoware_y = carla_location.y 
        
        autoware_yaw = carla_rotation.yaw
        
        rotation = Rotation.from_euler('z', autoware_yaw, degrees=True)
        quaternion = rotation.as_quat()
        
        autoware_pose = {
            'x': autoware_x,
            'y': autoware_y,
            'z': carla_location.z,  # Assuming Z remains the same
            'x1': 0.0,
            'y1': 0.0,
            'z1': quaternion[3],
            'w': quaternion[2] * -1.0
        }
        self.pose_msg = autoware_pose

    #rclpy.init(args=args)
    #pose_publisher = PosePublisher()

    #try:
    #    rclpy.spin(pose_publisher)
    #except KeyboardInterrupt:
    #    pass
    #finally:
    #    pose_publisher.destroy_node()
    #    rclpy.shutdown()

