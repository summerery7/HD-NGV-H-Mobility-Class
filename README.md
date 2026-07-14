# 현대 NGV H-모빌리티 클래스 심화교육

MG400 로봇 + MuJoCo + ROS2 + 비전 5일 과정 실습 코드입니다.

| 일차 | 내용 | 폴더 |
|---|---|---|
| day_1 | ROS2 튜토리얼(turtlesim), MuJoCo MG400 모델 로드 + 임의 토크 인가 | `day_1/` |
| day_2 | MuJoCo 로봇 제어 — FK/IK/자코비안/동역학/파라미터 추정/CTM 제어 | `day_2/` |
| day_3 | ROS2 연동 (`mujoco_ros` / `realworld_ros` 패키지) + YOLO 픽셀 검출 | `day_3/` |
| day_4 | 카메라 캘리브레이션, 로봇-아이 캘리브레이션, 3D Localization, YOLO 3D 좌표 | `day_4/` |
| day_5 | Visual Servoing (IBVS/PBVS) + 통합 프로젝트 패키지 틀 (`final_project`) | `day_5/` |

## 환경 설정

권장 환경: **Ubuntu 22.04 + ROS 2 Humble + Python 3.10**

### 1. 파이썬 라이브러리 설치

```bash
pip install -r requirements.txt
```

day_1 ~ day_5 실습에 필요한 패키지(mujoco, opencv-python, torch, open3d, PyQt5, pyrealsense2 등)가 모두 설치됩니다.

- **GPU(CUDA)로 YOLO 추론을 하시는 경우**: [pytorch.org](https://pytorch.org/get-started/locally/) 안내에 따라 CUDA 버전에 맞는 `torch`/`torchvision`을 먼저 설치하신 뒤 `pip install -r requirements.txt`를 실행해 주세요.
- YOLO는 `torch.hub.load("ultralytics/yolov5", ...)` 방식으로 로드하므로 최초 실행 시 인터넷 연결이 필요합니다.

### 2. ROS 2 (day_1, day_3, day_5)

`rclpy`, `turtlesim` 등 ROS 관련 모듈은 pip이 아니라 ROS 2 Humble 설치 시 함께 제공됩니다.

```bash
sudo apt install ros-humble-desktop ros-humble-turtlesim python3-tk
```

`dobot_msgs`는 교육용 커스텀 메시지 패키지로, 워크스페이스에서 `colcon build`로 빌드합니다.

## day_1

- `ros_tutorial/` — turtlesim 실습 코드입니다. (학생들이 직접 ROS2 패키지로 구성합니다.)
- `mujoco/load_mg400.py` — MG400 모델을 MuJoCo 뷰어에 로드합니다.
- `mujoco/apply_random_torque.py` — 관절에 임의 토크를 인가합니다.

## day_2

`1.Forward_Kinematics` ~ `9.Sympy_example` 순서로 진행합니다.
각 스크립트는 폴더 구조 그대로 실행해 주세요. (`8.xml`의 모델과 `7.utils`를 상대경로로 참조합니다.)

## day_3

colcon 빌드 및 `ros2 run` 실행 방법은 `day_3/README.md`를 참고해 주세요.

## day_4

- `1.camera_calibration/calib_pose.py` — 카메라 캘리브레이션 GUI입니다.
- `2.robot_eye_calibration/robot_eye_calibration.py` — 로봇-아이(6DOF) 캘리브레이션입니다.
- `3.3d_localization/chair_localization.py` — 포인트클라우드 3D Localization입니다.
- `4.yolo_locate/yolo_locate.py` — YOLO 검출 결과와 깊이 카메라로 X, Y, Z [mm] 좌표를 출력합니다.

## day_5

- `visual_servoing/ibvs.py`, `pbvs.py` — 비주얼 서보잉 실습 코드입니다. (ROS를 사용하지 않습니다.)
- `final_project/` — YOLO+CAMERA+ROBOT+ROS 통합 프로젝트용 ROS2 패키지 틀입니다.
  (노드 이름만 있는 빈 파일이며, 내용은 학생들이 직접 작성합니다.)

```bash
ros2 run final_project robot_control
ros2 run final_project yolo_detector
ros2 run final_project camera_calibration
ros2 run final_project main_controller
```

## 참고

- YOLO 가중치(`*.pt`)는 저장소에 포함되어 있지 않습니다. 각자 학습한 가중치를 사용해 주세요.
- ROS2 패키지 빌드는 `colcon build` 후 `source install/setup.bash`를 실행해 주세요.
