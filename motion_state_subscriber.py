import rclpy
from rclpy.node import Node
from autoware_adapi_v1_msgs.msg import MotionState  # Ensure this message type is available in your workspace

class MotionStateSubscriber(Node):
    def __init__(self, world):
        super().__init__('motion_state_subscriber')
        # Create a subscription to the /api/motion/state topic
        self.subscription = self.create_subscription(
            MotionState,
            '/api/motion/state',
            self.motion_state_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.world = world
        self.stopped = False
        self.get_logger().info('Subscribed to /api/motion/state topic')

    def motion_state_callback(self, msg):
        # Check if the vehicle is stopped
        if msg.state == MotionState.STOPPED:  # Replace with the correct field if needed
            print('Vehicle is stopped')
            self.stopped = True
        else:
            print(f'Vehicle state: {msg.state}')
        self.world.tick()
        time.sleep(.1)

    def wait_until_stopped(self):
        # Keep spinning until the vehicle state is 'STOPPED'
        while not self.stopped:
            rclpy.spin_once(self)
