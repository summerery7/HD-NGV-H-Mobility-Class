import rclpy
from rclpy.node import Node
from turtlesim.srv import TeleportAbsolute

class TurtlesimAbsoluteClient(Node) :
    def __init__(self):
        super().__init__('turtlesim_abs_client')

        #--TODO--#

        while not self.client.wait_for_service(timeout_sec=1.0):
              self.get_logger().info('service not available, waiting again...')

        #--TODO--#

    def send_request(self, x, y, theta) :
          self.req.x = x
          self.req.y = y
          self.req.theta = theta
          #--TODO--#
          rclpy.spin_until_future_complete(self, self.future)
          #--TODO--#
    


def main(args=None):
        rclpy.init(args=args)

        client = TurtlesimAbsoluteClient()
        client.send_request(10.0 , 10.0,  1.0)

        client.destroy_node()
        rclpy.shutdown()

       
if __name__=='__main__' :
        main()
