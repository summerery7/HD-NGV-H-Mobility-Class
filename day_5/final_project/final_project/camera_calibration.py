#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_calibration.py  —  Camera 노드 (2/2)

    카메라 좌표계 물체 위치  --[고정 동차변환행렬 T_base_cam]-->  로봇 기준 물체 위치

        | Xb |   | 0  1  0   313              | | Xc |
        | Yb | = | 1  0  0   -12              | | Yc |
        | Zb |   | 0  0 -1   Z_ROBOT_TO_CAMERA| | Zc |
        | 1  |   | 0  0  0   1                | | 1  |

      Xb =  Yc + 313
      Yb =  Xc - 12
      Zb = -Zc + Z_ROBOT_TO_CAMERA          <-- 항상 같은 상수가 되어야 함

    마그네틱 그리퍼로 pick & place 하므로 로봇의 Z 는 절대 변하지 않는다.
    따라서 Zc(카메라~물체 거리) 를 상수로 고정하여 Zb 도 항상 같은 상수로 발행한다.
    (depth 측정 노이즈/구멍에 의해 Z 가 흔들리는 것을 원천 차단)

  구독
    /vision/object_camera (Float64MultiArray) [Xc, Yc, Zc, conf, cls]  mm
    /vision/object_class  (String)
  발행
    /vision/object_base   (Float64MultiArray) [Xb, Yb, Zb, conf, cls]  mm
    /vision/object_base_class (String)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


# ===========================================================================
#  ▼▼▼  사용자 설정 상수  ▼▼▼
# ===========================================================================

# [1] 로봇 base 원점에서 카메라까지의 Z 거리 [mm]  (동차변환행렬 3행 4열 성분)
#
#     설정 방법
#       ① 자석이 물체를 집는 높이로 로봇을 티칭하고 그때의 dobot/pose Z 값을 읽는다  -> Z_PICK
#       ② 그 상태에서 yolo_detector 가 출력하는 카메라 Z 값을 읽는다                -> Z_CAM_TO_OBJECT
#       ③ Z_ROBOT_TO_CAMERA = Z_PICK + Z_CAM_TO_OBJECT
#          (Zb = -Zc + Z_ROBOT_TO_CAMERA = Z_PICK 이 되도록 맞추는 것)
Z_ROBOT_TO_CAMERA = 250.0

# [2] 카메라 ~ 물체 윗면 거리 [mm]  (고정 상수)
#     None 으로 두면 yolo_detector 가 측정한 depth 를 그대로 사용한다.
#     마그네틱 pick & place 에서는 반드시 숫자로 고정할 것.
Z_CAM_TO_OBJECT = 600.0

# [3] 동차변환행렬 T_base_cam (로봇 base <- 카메라)
T_BASE_CAM = np.array([
    [0.0, 1.0,  0.0, 264.0],
    [1.0, 0.0,  0.0, -2.0],
    [0.0, 0.0, -1.0, Z_ROBOT_TO_CAMERA],
    [0.0, 0.0,  0.0, 1.0],
], dtype=np.float64)

#  ▲▲▲  여기까지 수정  ▲▲▲
# ===========================================================================


def transform_point(p_cam, T=T_BASE_CAM):
    """카메라 좌표 [X, Y, Z] (mm) -> 로봇 base 좌표 [X, Y, Z] (mm)"""
    p = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
    return (T @ p)[:3]


class CameraCalibration(Node):

    def __init__(self):
        super().__init__('camera_calibration')

        self.T = T_BASE_CAM
        np.set_printoptions(precision=3, suppress=True)
        self.get_logger().info(f'T_base_cam =\n{self.T}')

        # 고정 Zc 를 쓰는 경우, 로봇 기준 Z 는 항상 이 값으로 고정된다
        if Z_CAM_TO_OBJECT is not None:
            self.z_base_fixed = float(-Z_CAM_TO_OBJECT + Z_ROBOT_TO_CAMERA)
            self.get_logger().info(
                'Zc 고정 = %.1f mm  ->  로봇 기준 Z = %.1f mm (항상 일정)'
                % (Z_CAM_TO_OBJECT, self.z_base_fixed))
            self.get_logger().info(
                'main_controller.py 의 Z_WORK 를 %.1f 로 맞추세요.'
                % self.z_base_fixed)
        else:
            self.z_base_fixed = None
            self.get_logger().warn(
                'Z_CAM_TO_OBJECT = None -> 측정 depth 사용. '
                'Z 가 매번 달라질 수 있습니다.')

        self.create_subscription(Float64MultiArray, '/vision/object_camera',
                                 self.cb_object, 10)
        self.create_subscription(String, '/vision/object_class',
                                 self.cb_class, 10)
        self.pub_base = self.create_publisher(Float64MultiArray,
                                              '/vision/object_base', 10)
        self.pub_class = self.create_publisher(String,
                                               '/vision/object_base_class', 10)

        self.last_class = ''
        self.last_log = 0.0
        self.get_logger().info('좌표 변환 시작 — /vision/object_base 로 발행')

    def cb_class(self, msg):
        self.last_class = msg.data

    def cb_object(self, msg):
        if len(msg.data) < 3:
            return

        Xc, Yc = float(msg.data[0]), float(msg.data[1])
        Zc_meas = float(msg.data[2])

        # Z 는 상수로 고정 (마그네틱 pick & place -> 높이 변화 없음)
        Zc = float(Z_CAM_TO_OBJECT) if Z_CAM_TO_OBJECT is not None else Zc_meas

        p_b = transform_point([Xc, Yc, Zc], self.T)

        out = Float64MultiArray()
        data = [float(p_b[0]), float(p_b[1]), float(p_b[2])]
        if len(msg.data) >= 5:
            data.append(float(msg.data[3]))     # conf
            data.append(float(msg.data[4]))     # class id
        out.data = data                         # 리스트를 완성한 뒤 한 번에 대입
        self.pub_base.publish(out)
        if self.last_class:
            self.pub_class.publish(String(data=self.last_class))

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log >= 1.0:
            self.last_log = now
            self.get_logger().info(
                '[%s] cam(%.1f, %.1f, %.1f / 측정 Zc=%.1f) -> base(%.1f, %.1f, %.1f) mm'
                % (self.last_class or '-', Xc, Yc, Zc, Zc_meas,
                   p_b[0], p_b[1], p_b[2]))


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
