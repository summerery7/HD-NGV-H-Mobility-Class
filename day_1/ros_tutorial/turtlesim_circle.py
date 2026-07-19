import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3

class TurtlesimCirclePublisher(Node) :
    def __init__(self):
        super().__init__('turtlesim_circle')

        #--TODO--#

    def timer_callback(self):
          
          #--TODO--#


def main(args=None):
        rclpy.init(args=args)

        publisher = TurtlesimCirclePublisher()
        rclpy.spin(publisher)

        publisher.destroy_node()
        rclpy.shutdown()

if __name__=='__main__' :
        main()
