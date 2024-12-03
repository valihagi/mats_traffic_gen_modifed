import rclpy
from rclpy.node import Node
from autoware_adapi_v1_msgs.srv import ChangeOperationMode

class AutonomousModeClient(Node):
    def __init__(self):
        super().__init__('autonomous_mode_client')
        # Create a client for the change_to_autonomous service
        self.client = self.create_client(ChangeOperationMode, '/api/operation_mode/change_to_autonomous')
        
        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /api/operation_mode/change_to_autonomous service...')


    def send_request(self):
        # Prepare a request (empty as per your example)
        request = ChangeOperationMode.Request()
        
        # Call the service and handle the result
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info('Autonomous mode enabled successfully')
            return 0
        else:
            self.get_logger().error('Failed to enable autonomous mode')
            return 1


