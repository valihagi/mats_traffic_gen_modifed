import rclpy
from rclpy.node import Node
from autoware_adapi_v1_msgs.srv import ChangeOperationMode

class ChangeOperationModeClient(Node):
    def __init__(self):
        super().__init__('change_operation_mode_client')
        # Create a service client
        self.client = self.create_client(ChangeOperationMode, '/api/operation_mode/change_to_autonomous')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def send_request(self):
        # Create a request message
        request = ChangeOperationMode.Request()
        
        # Set the desired operation mode
        request.mode = ChangeOperationMode.Request.AUTONOMOUS  # Adjust as per your message definition

        # Call the service and wait for the response
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # Check the result
        if future.result() is not None:
            self.get_logger().info('Service response: {}'.format(future.result()))
        else:
            self.get_logger().error('Service call failed: {}'.format(future.exception()))


