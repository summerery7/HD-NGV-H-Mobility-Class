# 현대 NGV H-모빌리티 클래스 심화교육

MG400 로봇 + MuJoCo + ROS2 + 비전 5일 과정 실습 코드.

| 일차 | 내용 | 폴더 |
|---|---|---|
| day_1 | ROS2 튜토리얼(turtlesim), MuJoCo MG400 모델 로드 + 임의 토크 인가 | `day_1/` |
| day_2 | MuJoCo 로봇 제어 — FK/IK/자코비안/동역학/파라미터 추정/CTM 제어 | `day_2/` |
| day_3 | ROS2 연동 (`mujoco_ros` / `realworld_ros` 패키지) + YOLO 픽셀 검출 | `day_3/` |
| day_4 | 카메라 캘리브레이션, 로봇-아이 캘리브레이션, 3D Localization, YOLO 3D 좌표 | `day_4/` |
| day_5 | Visual Servoing (IBVS/PBVS) + 통합 프로젝트 패키지 틀 (`final_project`) | `day_5/` |

## day_1

- `ros_tutorial/` — turtlesim 실습 코드 (학생들이 직접 ROS2 패키지로 구성)
- `mujoco/load_mg400.py` — MG400 모델을 MuJoCo 뷰어에 로드
- `mujoco/apply_random_torque.py` — 관절에 임의 토크 인가

## day_2

`1.Forward_Kinematics` ~ `8.xml`, `9.Sympy_example` 순서로 진행.
각 스크립트는 폴더 구조 그대로 실행 (`8.xml`의 모델과 `7.utils`를 상대경로로 참조).

## day_3

`day_3/README.md` 참고 — colcon 빌드 및 `ros2 run` 실행 방법.

## day_4

- `1.camera_calibration/calib_pose.py` — 카메라 캘리브레이션 GUI
- `2.robot_eye_calibration/robot_eye_calibration.py` — 로봇-아이(6DOF) 캘리브레이션
- `3.3d_localization/chair_localization.py` — 포인트클라우드 3D Localization
- `4.yolo_locate/yolo_locate.py` — YOLO 검출 + 깊이 카메라로 X, Y, Z [mm] 좌표 출력

## day_5

- `visual_servoing/ibvs.py`, `pbvs.py` — 비주얼 서보잉 (ROS 미사용)
- `final_project/` — YOLO+CAMERA+ROBOT+ROS 통합 프로젝트용 ROS2 패키지 틀
  (노드 이름만 있는 빈 파일 — 내용은 학생들이 작성)

```bash
ros2 run final_project robot_control
ros2 run final_project yolo_detector
ros2 run final_project camera_calibration
ros2 run final_project main_controller
```

## 참고

- YOLO 가중치(`*.pt`)는 저장소에 포함하지 않는다 — 각자 학습한 가중치 사용.
- ROS2 패키지 빌드: `colcon build` 후 `source install/setup.bash`.
