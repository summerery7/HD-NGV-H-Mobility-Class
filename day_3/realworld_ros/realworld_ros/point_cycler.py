# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from dobot_msgs.msg import PoseCmd
from dobot_msgs.msg import JointCmd


POINTS = [
    [210.0, -1.0, 0.0, 15.0],
    [210.0, 59.0, 0.0, 15.0],
    [250.0, 29.0, 30.0, 15.0],
]

SPEED = 100.0
TOLERANCE = 1.0


class PointCycler(Node):

    def __init__(self):
        super().__init__('point_cycler')
        self.pub_cmd = self.create_publisher(PoseCmd, 'dobot/cmd/mov_j', 10)
        self.create_subscription(Float64MultiArray, 'dobot/pose', self.cb_pose, 10)

        self.index = 0
        self.target = None
        self.lap = 0

        self.get_logger().info(f'포인트 {len(POINTS)}개 순환 시작 (속도 {SPEED} deg/s)')

    def send_next(self):
        point = POINTS[self.index]
        msg = PoseCmd()
        msg.x, msg.y, msg.z, msg.r = point
        msg.speed = SPEED
        self.pub_cmd.publish(msg)
        self.target = point
        self.get_logger().info(f'[{self.lap}바퀴] P{self.index + 1} {point} 로 이동 명령')

    def cb_pose(self, msg):
        pose = msg.data
        if self.target is None:

            self.send_next()
            return

        if all(abs(pose[i] - self.target[i]) <= TOLERANCE for i in range(4)):
            self.index = (self.index + 1) % len(POINTS)
            if self.index == 0:
                self.lap += 1
            self.send_next()


def main(args=None):
    rclpy.init(args=args)
    node = PointCycler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
