# day_3 — ROS2 연동 (MuJoCo / 실제 로봇) + YOLO

## 구성

| 폴더 | 내용 |
|---|---|
| `mujoco_ros/` | ROS2 패키지 — MuJoCo 시뮬레이션 연동 (CTM 제어기 노드 + 데모 명령 노드) |
| `realworld_ros/` | ROS2 패키지 — 실제 로봇(Dobot MG400) 연동 |
| `dobot_msgs/` | 커스텀 메시지 패키지 (JointCmd, PoseCmd) — 두 패키지 공용 |
| `yolo/` | YOLO 검출 + 중심 픽셀 (u, v) 출력 (ROS 미사용) |

## 빌드

```bash
cd ~/ros2_ws/src
cp -r <repo>/day_3/mujoco_ros <repo>/day_3/realworld_ros <repo>/day_3/dobot_msgs .
cd ~/ros2_ws
colcon build --packages-select dobot_msgs mujoco_ros realworld_ros
source install/setup.bash
```

## mujoco_ros — MuJoCo 연동

```bash
ros2 run mujoco_ros mujoco_controller     # 터미널 1: MuJoCo 뷰어 + CTM 제어기
ros2 run mujoco_ros command_sender        # 터미널 2: 데모 명령 전송
```

| 토픽 | 타입 | 의미 |
|---|---|---|
| `mujoco/cmd/joint` | `dobot_msgs/JointCmd` | 관절 목표 (deg), speed = deg/s |
| `mujoco/cmd/mov_l` | `dobot_msgs/PoseCmd` | 말단 직선 이동 (mm), speed = mm/s |
| `mujoco/joint_states` | `sensor_msgs/JointState` | 관절 위치/속도/토크 (100 Hz) |
| `mujoco/pose` | `std_msgs/Float64MultiArray` | 말단 [x, y, z, r] |
| `mujoco/moving` | `std_msgs/Bool` | 궤적 실행 중 여부 |

```bash
ros2 topic pub --once /mujoco/cmd/joint dobot_msgs/msg/JointCmd \
  "{j1: 20.0, j2: -10.0, j3: 10.0, j4: 45.0, speed: 40.0}"
```

## realworld_ros — 실제 로봇 연동

```bash
ros2 run realworld_ros dobot_control --ros-args -p ip:=192.168.1.6
ros2 run realworld_ros point_cycler       # 포인트 순환 데모
```

| 토픽 | 타입 | 의미 |
|---|---|---|
| `dobot/cmd/mov_j` | `dobot_msgs/PoseCmd` | 좌표 이동 (관절 보간), speed = deg/s |
| `dobot/cmd/mov_l` | `dobot_msgs/PoseCmd` | 좌표 이동 (직선 보간), speed = mm/s |
| `dobot/cmd/joint` | `dobot_msgs/JointCmd` | 관절 각도 이동, speed = deg/s |
| `dobot/pose` | `std_msgs/Float64MultiArray` | 현재 TCP 좌표 (10 Hz) |
| `dobot/joint_angle` | `std_msgs/Float64MultiArray` | 현재 관절 각도 (10 Hz) |
| `dobot/gripper` | `std_msgs/Bool` | 그리퍼 잡기/놓기 |
| `dobot/enable`, `dobot/clear_error`, ... | | 설정/제어 명령 (dobot_control.py 참고) |

## YOLO (픽셀 찍기)

```bash
cd day_3/yolo
python3 yolo_pixel.py --weights best.pt
```

검출된 물체의 바운딩 박스 중심 픽셀 (u, v)를 화면과 콘솔에 출력한다.
가중치(.pt)는 각자 학습한 파일을 사용한다.
