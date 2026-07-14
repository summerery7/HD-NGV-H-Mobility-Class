import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

class TurtlesimCircleSubscriber(Node) :
    def __init__(self):
        super().__init__('turtlesim_echo')

        self.subscriber = self.create_subscription(Twist, '/turtle1/cmd_vel', self.topic_callback, 10)

    def topic_callback(self,msg):
          self.get_logger().info(f'recv mes: {msg}')

    

def main(args=None):
        rclpy.init(args=args)

        subscriber = TurtlesimCircleSubscriber()
        rclpy.spin(subscriber)

        subscriber.destroy_node()
        rclpy.shutdown()

if __name__=='__main__' :
        main()
