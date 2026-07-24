#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robot_control.py  —  Dobot Controller 노드 진입점

    로봇 연결 및 초기화 / 위치에 따른 로봇 제어 / 로봇 현재 상태 발행 / 로봇 명령 구독
    → 이 기능은 이미 작성된 dobot_control.py 의 DobotControl 클래스가 모두 담당한다.
      따라서 새로 구현하지 않고 그대로 재사용한다.

  DobotControl 을 찾는 순서
    ① final_project 패키지 안 (dobot_control.py 를 복사해 둔 경우)
    ② realworld_ros 패키지 안 (기존 위치, 별도 복사 불필요)
    ③ 이 파일과 같은 폴더 (스크립트로 직접 실행할 때)

  ※ ②를 쓰려면 realworld_ros 가 빌드/설치되어 있어야 한다.
        colcon build --packages-select realworld_ros
        source install/setup.bash

  실행
    ros2 run final_project robot_control --ros-args -p ip:=192.168.1.6
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rclpy

DobotControl = None
_errors = []

# ① 같은 패키지 안
try:
    from .dobot_control import DobotControl
except Exception as e:
    _errors.append('final_project.dobot_control: %s' % e)

# ② 기존 realworld_ros 패키지 안
if DobotControl is None:
    try:
        from realworld_ros.dobot_control import DobotControl
    except Exception as e:
        _errors.append('realworld_ros.dobot_control: %s' % e)

# ③ 같은 폴더 (스크립트 직접 실행)
if DobotControl is None:
    try:
        from dobot_control import DobotControl
    except Exception as e:
        _errors.append('dobot_control: %s' % e)

if DobotControl is None:
    raise ImportError(
        '\n[DobotControl 을 찾을 수 없습니다]\n'
        '  시도한 경로:\n    - ' + '\n    - '.join(_errors) + '\n'
        '  해결 방법 (둘 중 하나)\n'
        '   1) realworld_ros 를 빌드/소스한다\n'
        '        colcon build --packages-select realworld_ros\n'
        '        source install/setup.bash\n'
        '   2) dobot_control.py 와 dobot_api.py 를 아래 폴더에 복사한다\n'
        '        %s' % SCRIPT_DIR)


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
