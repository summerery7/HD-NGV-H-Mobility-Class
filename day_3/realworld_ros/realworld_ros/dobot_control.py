# -*- coding: utf-8 -*-
import os
import sys
import threading
from time import sleep

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int32, String, Float64MultiArray, Int32MultiArray
from dobot_msgs.msg import PoseCmd, JointCmd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dobot_api import DobotApiDashboard, DobotApiMove, DobotApi, MyType


class DobotControl(Node):

    def __init__(self):
        super().__init__('dobot_control')


        self.declare_parameter('ip', '192.168.1.6')

        self.declare_parameter('joint_speed_max', 300.0)
        self.declare_parameter('linear_speed_max', 964.0)
        self.declare_parameter('speed_factor', 100)
        self.declare_parameter('gripper_do_index', 1)
        ip = self.get_parameter('ip').get_parameter_value().string_value
        self.joint_speed_max = self.get_parameter('joint_speed_max').value
        self.linear_speed_max = self.get_parameter('linear_speed_max').value
        self.speed_factor = self.get_parameter('speed_factor').value
        self.gripper_do_index = self.get_parameter('gripper_do_index').value


        self.get_logger().info(f'로봇({ip}) 접속 중...')
        self.dashboard = DobotApiDashboard(ip, 29999)
        self.move = DobotApiMove(ip, 30003)
        self.feed = DobotApi(ip, 30004)
        self.get_logger().info('접속 성공!')


        self.dashboard.ClearError()
        self.dashboard.EnableRobot()

        self.dashboard.SpeedFactor(self.speed_factor)
        self.get_logger().info(f'로봇 활성화 완료 (SpeedFactor={self.speed_factor})')


        self.feed_lock = threading.Lock()
        self.current_pose = None
        self.current_angle = None
        self.error_state = False
        feed_thread = threading.Thread(target=self.get_feed, daemon=True)
        feed_thread.start()


        self.create_subscription(PoseCmd, 'dobot/cmd/mov_j', self.cb_cmd_mov_j, 10)
        self.create_subscription(PoseCmd, 'dobot/cmd/mov_l', self.cb_cmd_mov_l, 10)
        self.create_subscription(JointCmd, 'dobot/cmd/joint', self.cb_cmd_joint, 10)


        self.create_subscription(Float64MultiArray, 'dobot/mov_j', self.cb_mov_j, 10)
        self.create_subscription(Float64MultiArray, 'dobot/mov_l', self.cb_mov_l, 10)
        self.create_subscription(Float64MultiArray, 'dobot/joint_mov_j', self.cb_joint_mov_j, 10)
        self.create_subscription(Float64MultiArray, 'dobot/rel_mov_j', self.cb_rel_mov_j, 10)
        self.create_subscription(Float64MultiArray, 'dobot/rel_mov_l', self.cb_rel_mov_l, 10)
        self.create_subscription(Float64MultiArray, 'dobot/arc', self.cb_arc, 10)
        self.create_subscription(Float64MultiArray, 'dobot/circle', self.cb_circle, 10)
        self.create_subscription(String, 'dobot/move_jog', self.cb_move_jog, 10)


        self.create_subscription(Bool, 'dobot/enable', self.cb_enable, 10)
        self.create_subscription(Empty, 'dobot/clear_error', self.cb_clear_error, 10)
        self.create_subscription(Empty, 'dobot/reset', self.cb_reset, 10)
        self.create_subscription(Empty, 'dobot/emergency_stop', self.cb_emergency_stop, 10)
        self.create_subscription(Int32, 'dobot/speed_factor', self.cb_speed_factor, 10)
        self.create_subscription(Int32, 'dobot/speed_j', self.cb_speed_j, 10)
        self.create_subscription(Int32, 'dobot/speed_l', self.cb_speed_l, 10)
        self.create_subscription(Int32, 'dobot/acc_j', self.cb_acc_j, 10)
        self.create_subscription(Int32, 'dobot/acc_l', self.cb_acc_l, 10)
        self.create_subscription(Int32, 'dobot/cp', self.cb_cp, 10)
        self.create_subscription(Int32MultiArray, 'dobot/do', self.cb_do, 10)
        self.create_subscription(Int32MultiArray, 'dobot/tool_do', self.cb_tool_do, 10)
        self.create_subscription(Bool, 'dobot/drag', self.cb_drag, 10)
        self.create_subscription(Bool, 'dobot/gripper', self.cb_gripper, 10)


        self.pub_pose = self.create_publisher(Float64MultiArray, 'dobot/pose', 10)
        self.pub_angle = self.create_publisher(Float64MultiArray, 'dobot/joint_angle', 10)
        self.pub_error = self.create_publisher(Bool, 'dobot/robot_error', 10)
        self.create_timer(0.1, self.publish_status)

        self.get_logger().info('토픽 준비 완료 — 명령 대기 중')


    def get_feed(self):
        hasRead = 0
        while True:
            data = bytes()
            while hasRead < 1440:
                temp = self.feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0
            feedInfo = np.frombuffer(data, dtype=MyType)
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with self.feed_lock:
                    self.current_pose = feedInfo['tool_vector_actual'][0]
                    self.current_angle = feedInfo['q_actual'][0]
                    self.error_state = bool(feedInfo['ErrorStatus'][0][0])
            sleep(0.001)

    def publish_status(self):
        with self.feed_lock:
            pose = self.current_pose
            angle = self.current_angle
            error = self.error_state
        if pose is not None:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in pose]
            self.pub_pose.publish(msg)
        if angle is not None:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in angle]
            self.pub_angle.publish(msg)
        self.pub_error.publish(Bool(data=error))


    def unit_to_percent(self, value, max_value, unit):
        percent = value / max_value * 100.0 / (self.speed_factor / 100.0)
        if percent > 100.0:
            achievable = max_value * self.speed_factor / 100.0
            self.get_logger().warn(
                f'요청 속도 {value}{unit}는 현재 SpeedFactor={self.speed_factor}에서 '
                f'최대 {achievable:.0f}{unit}를 초과 — 100%로 제한')
            percent = 100.0
        return max(1, round(percent))

    def speed_params(self, msg, speed_key, acc_key, max_value, unit):
        params = []
        if msg.speed > 0:
            pct = self.unit_to_percent(msg.speed, max_value, unit)
            params.append(f'{speed_key}={pct}')
        if msg.acc > 0:
            params.append(f'{acc_key}={msg.acc}')
        return params

    def cb_cmd_mov_j(self, msg):
        params = self.speed_params(msg, 'SpeedJ', 'AccJ', self.joint_speed_max, 'deg/s')
        self.get_logger().info(f'MovJ({msg.x}, {msg.y}, {msg.z}, {msg.r}) speed={msg.speed}deg/s → {params}')
        self.move.MovJ(msg.x, msg.y, msg.z, msg.r, *params)

    def cb_cmd_mov_l(self, msg):
        params = self.speed_params(msg, 'SpeedL', 'AccL', self.linear_speed_max, 'mm/s')
        self.get_logger().info(f'MovL({msg.x}, {msg.y}, {msg.z}, {msg.r}) speed={msg.speed}mm/s → {params}')
        self.move.MovL(msg.x, msg.y, msg.z, msg.r, *params)

    def cb_cmd_joint(self, msg):
        params = self.speed_params(msg, 'SpeedJ', 'AccJ', self.joint_speed_max, 'deg/s')
        self.get_logger().info(f'JointMovJ({msg.j1}, {msg.j2}, {msg.j3}, {msg.j4}) speed={msg.speed}deg/s → {params}')
        self.move.JointMovJ(msg.j1, msg.j2, msg.j3, msg.j4, *params)


    def check_len(self, msg, n, name):
        if len(msg.data) < n:
            self.get_logger().warn(f'{name}: 데이터 {n}개 필요 (받은 개수: {len(msg.data)})')
            return False
        return True

    def cb_mov_j(self, msg):
        if self.check_len(msg, 4, 'mov_j'):
            d = msg.data
            self.get_logger().info(f'MovJ({d[0]}, {d[1]}, {d[2]}, {d[3]})')
            self.move.MovJ(d[0], d[1], d[2], d[3])

    def cb_mov_l(self, msg):
        if self.check_len(msg, 4, 'mov_l'):
            d = msg.data
            self.get_logger().info(f'MovL({d[0]}, {d[1]}, {d[2]}, {d[3]})')
            self.move.MovL(d[0], d[1], d[2], d[3])

    def cb_joint_mov_j(self, msg):
        if self.check_len(msg, 4, 'joint_mov_j'):
            d = msg.data
            self.get_logger().info(f'JointMovJ({d[0]}, {d[1]}, {d[2]}, {d[3]})')
            self.move.JointMovJ(d[0], d[1], d[2], d[3])

    def cb_rel_mov_j(self, msg):
        if self.check_len(msg, 4, 'rel_mov_j'):
            d = msg.data
            self.get_logger().info(f'RelMovJ({d[0]}, {d[1]}, {d[2]}, {d[3]})')
            self.move.RelMovJ(d[0], d[1], d[2], d[3])

    def cb_rel_mov_l(self, msg):
        if self.check_len(msg, 4, 'rel_mov_l'):
            d = msg.data
            self.get_logger().info(f'RelMovL({d[0]}, {d[1]}, {d[2]}, {d[3]})')
            self.move.RelMovL(d[0], d[1], d[2], d[3])

    def cb_arc(self, msg):
        if self.check_len(msg, 8, 'arc'):
            d = msg.data
            self.get_logger().info(f'Arc(중간점 {d[0:4]}, 끝점 {d[4:8]})')
            self.move.Arc(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7])

    def cb_circle(self, msg):
        if self.check_len(msg, 9, 'circle'):
            d = msg.data
            self.get_logger().info(f'Circle(중간점 {d[0:4]}, 끝점 {d[4:8]}, {int(d[8])}바퀴)')
            self.move.Circle(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], int(d[8]))

    def cb_move_jog(self, msg):
        axis = msg.data.strip()
        self.get_logger().info(f'MoveJog({axis})')
        if axis:
            self.move.MoveJog(axis)
        else:
            self.move.MoveJog()


    def cb_enable(self, msg):
        if msg.data:
            self.get_logger().info('EnableRobot()')
            self.dashboard.EnableRobot()
        else:
            self.get_logger().info('DisableRobot()')
            self.dashboard.DisableRobot()

    def cb_clear_error(self, msg):
        self.get_logger().info('ClearError()')
        self.dashboard.ClearError()

    def cb_reset(self, msg):
        self.get_logger().info('ResetRobot()')
        self.dashboard.ResetRobot()

    def cb_emergency_stop(self, msg):
        self.get_logger().warn('EmergencyStop()!')
        self.dashboard.EmergencyStop()

    def cb_speed_factor(self, msg):
        self.get_logger().info(f'SpeedFactor({msg.data})')
        self.dashboard.SpeedFactor(msg.data)

        self.speed_factor = msg.data

    def cb_speed_j(self, msg):
        self.get_logger().info(f'SpeedJ({msg.data})')
        self.dashboard.SpeedJ(msg.data)

    def cb_speed_l(self, msg):
        self.get_logger().info(f'SpeedL({msg.data})')
        self.dashboard.SpeedL(msg.data)

    def cb_acc_j(self, msg):
        self.get_logger().info(f'AccJ({msg.data})')
        self.dashboard.AccJ(msg.data)

    def cb_acc_l(self, msg):
        self.get_logger().info(f'AccL({msg.data})')
        self.dashboard.AccL(msg.data)

    def cb_cp(self, msg):
        self.get_logger().info(f'CP({msg.data})')
        self.dashboard.CP(msg.data)

    def cb_do(self, msg):
        if self.check_len(msg, 2, 'do'):
            self.get_logger().info(f'DO({msg.data[0]}, {msg.data[1]})')
            self.dashboard.DO(msg.data[0], msg.data[1])

    def cb_tool_do(self, msg):
        if self.check_len(msg, 2, 'tool_do'):
            self.get_logger().info(f'ToolDO({msg.data[0]}, {msg.data[1]})')
            self.dashboard.ToolDO(msg.data[0], msg.data[1])

    def cb_gripper(self, msg):
        status = 1 if msg.data else 0
        self.get_logger().info(f'그리퍼 {"잡기" if status else "놓기"} — ToolDO({self.gripper_do_index}, {status})')
        self.dashboard.ToolDO(self.gripper_do_index, status)

    def cb_drag(self, msg):
        if msg.data:
            self.get_logger().info('StartDrag()')
            self.dashboard.StartDrag()
        else:
            self.get_logger().info('StopDrag()')
            self.dashboard.StopDrag()


    def shutdown_robot(self):
        try:
            self.dashboard.DisableRobot()
        except Exception:
            pass
        self.dashboard.close()
        self.move.close()
        self.feed.close()


def main(args=None):
    rclpy.init(args=args)
    node = DobotControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('종료 — 로봇 비활성화')
        node.shutdown_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
