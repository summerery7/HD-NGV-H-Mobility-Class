#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from dobot_msgs.msg import JointCmd, PoseCmd


SEQUENCE = [
    ("joint", (20.0, -10.0, 10.0, 45.0), 40.0),
    ("joint", (0.0, 0.0, 0.0, 0.0), 40.0),
    ("pose", (152.0, 0.0, 260.0, 0.0), 160.0),
    ("pose", (152.0, 60.0, 260.0, 0.0), 160.0),
    ("pose", (152.0, 60.0, 220.0, 0.0), 160.0),
    ("pose", (152.0, -60.0, 220.0, 0.0), 160.0),
    ("pose", (152.0, -60.0, 260.0, 0.0), 160.0),
    ("pose", (152.0, 0.0, 260.0, 0.0), 160.0),
]

MOVING_START_TIMEOUT = 5.0


class CommandSender(Node):
    def __init__(self):
        super().__init__("command_sender")
        self.pub_joint = self.create_publisher(JointCmd, "mujoco/cmd/joint", 10)
        self.pub_pose = self.create_publisher(PoseCmd, "mujoco/cmd/mov_l", 10)
        self.create_subscription(Bool, "mujoco/moving", self.cb_moving, 10)

        self.moving = None
        self.idx = 0
        self.done = False
        self.state = "SEND"
        self.t_mark = self.get_clock().now()
        self.timer = self.create_timer(0.05, self.step)

    def cb_moving(self, msg: Bool):
        self.moving = msg.data

    def _elapsed(self):
        return (self.get_clock().now() - self.t_mark).nanoseconds * 1e-9

    def step(self):
        if self.state == "SEND":
            if self.idx >= len(SEQUENCE):
                self.get_logger().info("모든 명령 완료 — 종료")
                self.timer.cancel()
                self.done = True
                return
            if (self.pub_joint.get_subscription_count() == 0
                    and self.pub_pose.get_subscription_count() == 0):
                if self._elapsed() > 2.0:
                    self.get_logger().info("mujoco_controller 대기 중...")
                    self.t_mark = self.get_clock().now()
                return
            kind, target, speed = SEQUENCE[self.idx]
            if kind == "joint":
                msg = JointCmd()
                msg.j1, msg.j2, msg.j3, msg.j4 = target
                msg.speed = speed
                self.pub_joint.publish(msg)
            else:
                msg = PoseCmd()
                msg.x, msg.y, msg.z, msg.r = target
                msg.speed = speed
                self.pub_pose.publish(msg)
            self.get_logger().info(
                f"[{self.idx + 1}/{len(SEQUENCE)}] {kind} {target} 전송")
            self.t_mark = self.get_clock().now()
            self.state = "WAIT_START"

        elif self.state == "WAIT_START":
            if self.moving:
                self.state = "WAIT_END"
            elif self._elapsed() > MOVING_START_TIMEOUT:
                self.get_logger().warn("moving 신호 없음 — 다음 명령으로 진행")
                self.idx += 1
                self.state = "SEND"

        elif self.state == "WAIT_END":
            if self.moving is False:
                self.idx += 1
                self.t_mark = self.get_clock().now()
                self.state = "SETTLE"

        elif self.state == "SETTLE":
            if self._elapsed() > 0.3:
                self.state = "SEND"


def main():
    rclpy.init()
    node = CommandSender()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
