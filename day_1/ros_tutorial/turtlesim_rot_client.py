import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from turtlesim.action import RotateAbsolute

class TurtlesimRotateClient(Node):
    def __init__(self):
        super().__init__('turtlesim_rot_client')

        #--TODO--#
        self.get_logger().info('action client started...')

    def send_goal(self,theta):
        
        #--TODO--#

        if self.client.wait_for_server(10) is False:
            self.get_logger().info('service not available...')
            return
        
        #--TODO--#

    def goal_callback(self, future):
        #--TODO--#

        if not self.goal_handle.accepted:
            self.get_logger().info('gole rejected ...')
            return
        
        self.get_logger().info('gole accepted...')
        #--TODO--#

    def feedback_callback(self,msg):
        #--TODO--#
        self.get_logger().info(f'recv feedback: {feedback.remaining}')

    def result_callback(self, future):
        #--TODO--#
        self.get_logger().info(f'recv result: {res.delta}')

def main(args=None):
    rclpy.init(args=args)

    client = TurtlesimRotateClient()
    client.send_goal(3.14)

    rclpy.spin(client)

if __name__ == '__main__':
    main()


