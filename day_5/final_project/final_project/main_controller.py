#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_controller.py  —  Main NODE

    /vision/object_base (로봇 기준 물체 위치) 구독
      -> mov_l 시퀀스 생성 -> dobot/cmd/mov_l, dobot/gripper 발행
      -> dobot/pose 로 도착 판정

  이동 경로 (제어 방식은 기존과 동일하게 mov_l 만 사용)
    바닥 높이에서 수평 이동하면 특이점/간섭으로 멈추므로,
    이동은 Z_TRAVEL(높은 위치)에서 하고 집고 놓을 때만 Z_WORK 로 내려온다.

    ① 물체 위로 이동 (x,  y,  Z_TRAVEL)
    ② 하강           (x,  y,  Z_WORK)     -> 완전 정지 -> 자석 ON  -> 대기
    ③ 상승           (x,  y,  Z_TRAVEL)
    ④ 배치 위로 이동 (px, py, Z_TRAVEL)
    ⑤ 하강           (px, py, Z_WORK)     -> 완전 정지 -> 자석 OFF -> 대기
    ⑥ 상승           (px, py, Z_TRAVEL)
    ⑦ 홈 복귀        (hx, hy, Z_TRAVEL)

  자석 동작 타이밍 (핵심)
    목표 좌표 근처에 들어온 것만으로는 도착으로 보지 않는다.
      · 허용오차 이내  +  직전 좌표와 거의 변화 없음(정지)  을
        ARRIVE_STABLE_N 회 연속 만족해야 '정지 완료' 로 인정
      · 정지 완료 후 PRE_GRIP_WAIT 만큼 더 기다렸다가 자석을 켠다
      · 자석 동작 후 POST_GRIP_WAIT 만큼 기다린 뒤에 다음 이동을 시작한다
    이렇게 해야 감속 중에 자석 명령이 나가서 늦게 붙는 문제가 생기지 않는다.
"""

import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from dobot_msgs.msg import PoseCmd


# ===========================================================================
#  ▼▼▼  사용자 설정 상수  ▼▼▼
# ===========================================================================

# [1] 자석이 물체에 닿는 높이 [mm]  (pick / place 할 때만 이 높이로 내려온다)
Z_WORK = -155.0

# [2] 이동할 때의 높이 [mm]  (반드시 Z_WORK 보다 높아야 함)
Z_TRAVEL = -60.0

# [3] 물체를 놓을 위치 [X, Y] (mm, 로봇 base 기준) — 임의로 지정
PLACE_POINT = [250.0, 29.0]

# [4] 작업 후 복귀할 대기 위치 [X, Y] (mm)
HOME_POINT = [210.0, -1.0]

# [5] 툴 회전각 R [deg] — 모든 이동에서 동일
R_ANGLE = 15.0

# [6] 이동 속도 [mm/s]
SPEED = 100.0          # 수평 이동
SPEED_Z = 50.0         # 상하 승강 (물체에 접근/이탈하므로 느리게)

# [7] 대상 클래스 이름 ('' 이면 검출된 모든 물체 대상)
TARGET_CLASS = ''

# ---- 도착/정지 판정 -------------------------------------------------------
TOLERANCE = 1.5        # XYZ 도착 허용오차 [mm]
R_TOLERANCE = 2.0      # R 도착 허용오차 [deg]
STOP_EPS = 0.3         # 직전 좌표 대비 변화가 이보다 작으면 '정지' 로 판단 [mm/deg]
ARRIVE_STABLE_N = 3    # 위 조건을 연속 몇 회 만족해야 도착으로 인정 (pose 는 10 Hz)

# ---- 자석 동작 타이밍 -----------------------------------------------------
PRE_GRIP_WAIT = 0.5    # 정지 확인 후 자석을 켜기까지 대기 [s]
POST_GRIP_WAIT = 1.0   # 자석 동작 후 다음 이동까지 대기 [s] (붙는 시간 확보)

# ---- 기타 타이밍 ----------------------------------------------------------
MOVE_TIMEOUT = 15.0    # 이동 타임아웃 [s]
COOLDOWN = 2.0         # 1회 작업 완료 후 재검출까지 대기 [s]

# ---- 목표 확정 조건 -------------------------------------------------------
N_SAMPLES = 10         # 검출 좌표 몇 개를 모아서 판단할지
STABLE_TOL = 5.0       # 샘플 흔들림 허용 [mm]

# ---- 안전 범위 (MG400 작업영역) -------------------------------------------
XY_MIN_RADIUS = 150.0  # base 로부터 최소 반경 [mm]
XY_MAX_RADIUS = 400.0  # base 로부터 최대 반경 [mm]
Z_CHECK_TOL = 5.0      # 수신 Z 와 Z_WORK 차이가 이보다 크면 경고

AUTO_START = True      # True 면 물체가 검출되는 즉시 자동 실행

#  ▲▲▲  여기까지 수정  ▲▲▲
# ===========================================================================


class MainController(Node):

    def __init__(self):
        super().__init__('main_controller')

        # ---- 통신 -----------------------------------------------------------
        self.pub_cmd = self.create_publisher(PoseCmd, 'dobot/cmd/mov_l', 10)
        self.pub_gripper = self.create_publisher(Bool, 'dobot/gripper', 10)

        self.create_subscription(Float64MultiArray, '/vision/object_base',
                                 self.cb_object, 10)
        self.create_subscription(String, '/vision/object_base_class',
                                 self.cb_class, 10)
        self.create_subscription(Float64MultiArray, 'dobot/pose', self.cb_pose, 10)
        self.create_subscription(Bool, 'dobot/robot_error', self.cb_error, 10)
        self.create_subscription(Bool, 'main/start', self.cb_start, 10)

        # ---- 상태 -----------------------------------------------------------
        # WAIT / MOVING / PRE_GRIP / POST_GRIP / COOLDOWN
        self.state = 'WAIT'
        self.plan = []               # [(이름, [x, y, z, r], 속도, 도착 후 그리퍼), ...]
        self.step = 0
        self.target = None
        self.pose = None
        self.prev_pose = None
        self.arrive_count = 0        # 연속 정지 확인 횟수
        self.pending_action = None   # 정지 후 실행할 자석 동작
        self.samples = deque(maxlen=N_SAMPLES)
        self.last_class = ''
        self.z_warned = False
        self.t_cmd = 0.0
        self.t_state = 0.0
        self.enabled = AUTO_START

        self.create_timer(0.1, self.watchdog)

        if Z_TRAVEL <= Z_WORK:
            self.get_logger().error(
                'Z_TRAVEL(%.1f) 이 Z_WORK(%.1f) 보다 낮습니다! 상수를 확인하세요.'
                % (Z_TRAVEL, Z_WORK))

        self.get_logger().info(
            'Main NODE 준비 — Z_WORK=%.1f / Z_TRAVEL=%.1f mm, PLACE=%s, HOME=%s'
            % (Z_WORK, Z_TRAVEL, PLACE_POINT, HOME_POINT))

    # =======================================================================
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # =======================================================================
    # 콜백
    # =======================================================================
    def cb_class(self, msg):
        self.last_class = msg.data

    def cb_start(self, msg):
        self.enabled = bool(msg.data)
        self.get_logger().info(f'수동 트리거: {"시작" if self.enabled else "정지"}')

    def cb_error(self, msg):
        if msg.data and self.state in ('MOVING', 'PRE_GRIP', 'POST_GRIP'):
            self.get_logger().error('로봇 에러 발생 — 시퀀스 중단')
            self.abort()

    def cb_pose(self, msg):
        if len(msg.data) < 4:
            return
        self.prev_pose, self.pose = self.pose, [float(v) for v in msg.data[:4]]

        if self.state != 'MOVING' or self.target is None:
            return

        # (1) 목표 허용오차 이내인가
        xyz_ok = all(abs(self.pose[i] - self.target[i]) <= TOLERANCE for i in range(3))
        r_ok = abs(self.pose[3] - self.target[3]) <= R_TOLERANCE

        # (2) 실제로 멈췄는가 (직전 피드백 대비 변화량)
        if self.prev_pose is None:
            moved = 1e9
        else:
            moved = max(abs(self.pose[i] - self.prev_pose[i]) for i in range(4))
        stopped = moved <= STOP_EPS

        if xyz_ok and r_ok and stopped:
            self.arrive_count += 1
            if self.arrive_count >= ARRIVE_STABLE_N:
                self.arrive_count = 0
                self.on_arrived()
        else:
            self.arrive_count = 0

    def cb_object(self, msg):
        """검출 좌표를 모아 흔들림이 작으면 목표로 확정하고 시퀀스 시작"""
        if self.state != 'WAIT' or not self.enabled or len(msg.data) < 3:
            return
        if TARGET_CLASS and self.last_class and self.last_class != TARGET_CLASS:
            return

        z_in = float(msg.data[2])
        if not self.z_warned and abs(z_in - Z_WORK) > Z_CHECK_TOL:
            self.z_warned = True
            self.get_logger().warn(
                '수신 Z=%.1f mm 가 Z_WORK=%.1f mm 와 다릅니다. '
                'camera_calibration.py 의 상수와 Z_WORK 를 일치시키세요. '
                '(이동은 Z_WORK 로 수행)' % (z_in, Z_WORK))

        self.samples.append(np.array(msg.data[0:2], dtype=np.float64))
        if len(self.samples) < N_SAMPLES:
            return

        arr = np.array(self.samples)
        spread = float(np.max(np.linalg.norm(arr - np.median(arr, axis=0), axis=1)))
        if spread > STABLE_TOL:
            self.samples.popleft()      # 아직 흔들림 -> 계속 관찰
            return

        obj_xy = np.median(arr, axis=0)
        self.samples.clear()
        self.start_pick(float(obj_xy[0]), float(obj_xy[1]))

    # =======================================================================
    # 시퀀스 생성 / 실행
    # =======================================================================
    def in_workspace(self, x, y):
        r = math.hypot(x, y)
        if not (XY_MIN_RADIUS <= r <= XY_MAX_RADIUS):
            self.get_logger().warn(
                '작업영역 밖: (%.1f, %.1f) 반경 %.1f mm (허용 %.0f~%.0f)'
                % (x, y, r, XY_MIN_RADIUS, XY_MAX_RADIUS))
            return False
        return True

    def start_pick(self, x, y):
        if not self.in_workspace(x, y):
            return

        px, py = PLACE_POINT
        hx, hy = HOME_POINT
        R = R_ANGLE

        #        단계 이름            목표 [x, y, z, r]           속도      도착 후 자석
        self.plan = [
            ('① 물체 위로 이동', [x,  y,  Z_TRAVEL, R], SPEED,   None),
            ('② 하강',           [x,  y,  Z_WORK,   R], SPEED_Z, True),   # 자석 ON
            ('③ 상승',           [x,  y,  Z_TRAVEL, R], SPEED_Z, None),
            ('④ 배치 위로 이동', [px, py, Z_TRAVEL, R], SPEED,   None),
            ('⑤ 하강',           [px, py, Z_WORK,   R], SPEED_Z, False),  # 자석 OFF
            ('⑥ 상승',           [px, py, Z_TRAVEL, R], SPEED_Z, None),
            ('⑦ 홈 복귀',        [hx, hy, Z_TRAVEL, R], SPEED,   None),
        ]
        self.step = 0
        self.get_logger().info(
            '물체 확정 [%s] X=%.1f Y=%.1f (pick Z=%.1f) — Pick & Place 시작'
            % (self.last_class or '-', x, y, Z_WORK))
        self.send_step()

    def send_step(self):
        name, point, speed, _ = self.plan[self.step]
        msg = PoseCmd()
        msg.x, msg.y, msg.z, msg.r = [float(v) for v in point]
        msg.speed = float(speed)
        msg.acc = 0            # PoseCmd.acc 는 int 타입 (0 이면 가속도 지정 안 함)
        self.pub_cmd.publish(msg)

        self.target = [float(v) for v in point]
        self.state = 'MOVING'
        self.arrive_count = 0
        self.t_cmd = self.now()
        self.get_logger().info(
            '%s -> (%.1f, %.1f, %.1f, %.1f) @ %.0f mm/s'
            % (name, point[0], point[1], point[2], point[3], speed))

    def set_gripper(self, on):
        self.pub_gripper.publish(Bool(data=bool(on)))
        self.get_logger().info(f'그리퍼(자석) {"ON (잡기)" if on else "OFF (놓기)"}')

    def on_arrived(self):
        """허용오차 이내 + 정지 상태가 연속 확인된 시점"""
        name, _, _, action = self.plan[self.step]
        self.get_logger().info(f'{name} 도착 (정지 확인)')

        self.pending_action = action
        if action is None:
            self.advance()                  # 자석 동작 없음 -> 바로 다음 단계
        else:
            self.state = 'PRE_GRIP'         # 완전히 멈출 시간을 조금 더 준 뒤 자석 동작
            self.t_state = self.now()

    def advance(self):
        self.step += 1
        if self.step >= len(self.plan):
            self.finish()
        else:
            self.send_step()

    def finish(self):
        self.target = None
        self.plan = []
        self.pending_action = None
        self.state = 'COOLDOWN'
        self.t_state = self.now()
        self.get_logger().info('Pick & Place 완료 — 다음 물체 대기')

    def abort(self):
        self.set_gripper(False)
        self.target = None
        self.plan = []
        self.pending_action = None
        self.samples.clear()
        self.state = 'COOLDOWN'
        self.t_state = self.now()

    # =======================================================================
    def watchdog(self):
        t = self.now()

        # 정지 확인 후 -> 자석 동작
        if self.state == 'PRE_GRIP' and t - self.t_state >= PRE_GRIP_WAIT:
            self.set_gripper(self.pending_action)
            self.state = 'POST_GRIP'
            self.t_state = t

        # 자석이 붙을(떨어질) 시간을 준 뒤 -> 다음 이동
        elif self.state == 'POST_GRIP' and t - self.t_state >= POST_GRIP_WAIT:
            self.pending_action = None
            self.advance()

        elif self.state == 'COOLDOWN' and t - self.t_state >= COOLDOWN:
            self.state = 'WAIT'
            self.samples.clear()

        elif self.state == 'MOVING' and t - self.t_cmd >= MOVE_TIMEOUT:
            self.get_logger().error(
                '이동 타임아웃(%.0fs) — 시퀀스 중단. 목표=%s, 현재=%s'
                % (MOVE_TIMEOUT, self.target, self.pose))
            self.abort()


def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.set_gripper(False)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
