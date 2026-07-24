#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_detector.py  —  Camera 노드 (1/2)

    [YOLO best.pt]              -> 물체 픽셀 (u, v)
    [RealSense depth + 캘리브]  -> 카메라 좌표계 3D 위치 (X, Y, Z) [mm]

  발행 토픽
    /vision/object_camera  (Float64MultiArray) [X, Y, Z, conf, cls_id]  단위 mm
    /vision/object_class   (String)            검출 클래스 이름

  실행 예
    ros2 run <패키지명> yolo_detector --ros-args \
        -p weights:=/home/user/best.pt \
        -p params_xml:=/home/user/camera_params.xml \
        -p target_class:=square

  ※ 기존 yolo_locate_functions.py 가 같은 폴더/패키지에 있으면 그대로 재사용하고,
     없으면 이 파일 하단의 내장 구현(_fallback)을 사용한다.
"""

import os
import sys
import time
import threading

import numpy as np
import cv2
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# 기존 yolo_locate_functions.py 재사용 (없으면 내장 구현 사용)
# ---------------------------------------------------------------------------
try:
    from yolo_locate_functions import (
        load_xml_camera_matrix_dist,
        load_factory_extrinsic,
        load_yolo_model,
        DetectorThread,
        box_depth_mm,
        sample_depth_mm,
        deproject_pixel,
        color_to_depth_frame,
        draw_detection,
    )
    _HELPER_SRC = "yolo_locate_functions.py"
except ImportError:
    _HELPER_SRC = "내장 구현(fallback)"

    # ----- 캘리브레이션 XML -------------------------------------------------
    def load_xml_camera_matrix_dist(path):
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()

        def _vec(tag):
            node = root.find(tag)
            if node is None:
                raise ValueError("XML 에 <%s> 없음" % tag)
            return [float(c.text) for c in list(node)]

        K = np.array(_vec("camera_matrix"), dtype=np.float64).reshape(3, 3)
        dist = np.array(_vec("camera_distortion"), dtype=np.float64).reshape(1, -1)
        return K, dist

    # ----- color -> depth 외부 파라미터 (공장 출하값) -----------------------
    def load_factory_extrinsic(csp, dsp):
        ext = csp.get_extrinsics_to(dsp)
        # librealsense 회전행렬은 column-major 이므로 전치
        R = np.array(ext.rotation, dtype=np.float64).reshape(3, 3).T
        t_mm = np.array(ext.translation, dtype=np.float64) * 1000.0
        return R, t_mm, "factory"

    # ----- YOLOv5 모델 로드 -------------------------------------------------
    def load_yolo_model(weights, conf=0.25, iou=0.45, max_det=20,
                        repo="ultralytics/yolov5", source="github", half=True,
                        yolov5_dir=None):
        # 경로 검사를 먼저 한다.
        # (없는 경로를 그대로 넘기면 yolov5 내부에서 '.' 로 바뀌어
        #  "Is a directory: '.'" 라는 엉뚱한 에러가 난다)
        weights = os.path.abspath(os.path.expanduser(str(weights)))
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                "\n[가중치 파일 없음] %s\n"
                "  ros2 run 은 install/ 폴더를 실행하므로 소스 폴더의 best.pt 는 보이지 않습니다.\n"
                "  아래처럼 절대경로로 지정하세요.\n"
                "    ros2 run <패키지> yolo_detector --ros-args -p weights:=/절대/경로/best.pt\n"
                "  경로 찾기: find ~/workspace/ros_ws/src -name 'best.pt'" % weights)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # yolov5 저장소를 찾아 직접 로드한다.
        # (torch.hub.load 는 torch/yolov5 버전 조합에 따라 path 인자가 유실되어
        #  "Is a directory: '.'" 에러가 나는 경우가 있어 우회한다)
        candidates = [d for d in [
            yolov5_dir,
            os.path.expanduser("~/.cache/torch/hub/ultralytics_yolov5_master"),
            os.path.expanduser("~/yolov5"),
        ] if d]
        repo_dir = next(
            (d for d in candidates
             if os.path.isfile(os.path.join(os.path.expanduser(d), "hubconf.py"))),
            None)

        model = None
        if repo_dir:
            repo_dir = os.path.expanduser(repo_dir)
            sys.path.insert(0, repo_dir)
            try:
                from models.common import AutoShape, DetectMultiBackend
                from utils.torch_utils import select_device
                dev = select_device("" if device == "cuda" else "cpu")
                model = AutoShape(DetectMultiBackend(weights, device=dev, fuse=True))
                print("[yolo] 로컬 저장소에서 직접 로드: %s" % repo_dir)
            except Exception as e:
                print("[yolo] 직접 로드 실패(%s) -> torch.hub 로 재시도" % e)
                model = None
            finally:
                # yolov5 의 models/utils 패키지가 다른 모듈을 가리는 것을 막기 위해
                # 로드가 끝나면 sys.path 에서 제거한다 (이미 import 된 것은 유지됨)
                if repo_dir in sys.path:
                    sys.path.remove(repo_dir)

        if model is None:
            model = torch.hub.load(repo, "custom", path=weights, source=source)

        model.conf, model.iou, model.max_det = conf, iou, max_det
        try:
            model.to(device)
            if half and device == "cuda":
                model.half()
        except Exception:
            pass
        model.eval()
        names = model.names if isinstance(model.names, (list, tuple)) \
            else [model.names[i] for i in sorted(model.names)]
        return model, names, device

    # ----- 추론 전용 스레드 -------------------------------------------------
    class DetectorThread(threading.Thread):
        """최신 프레임 1장만 유지하며 비동기 추론 (디스플레이 프레임 저하 방지)"""

        def __init__(self, model, imgsz=640):
            super().__init__(daemon=True)
            self.model, self.imgsz = model, imgsz
            self._in, self._out = None, None
            self._lock = threading.Lock()
            self._evt = threading.Event()
            self._run = True
            self.infer_ms = 0.0

        def submit(self, rgb):
            with self._lock:
                self._in = rgb
            self._evt.set()

        def get(self):
            with self._lock:
                return self._out

        def run(self):
            while self._run:
                self._evt.wait(0.1)
                self._evt.clear()
                with self._lock:
                    img = self._in
                    self._in = None
                if img is None:
                    continue
                t0 = time.time()
                res = self.model(img, size=self.imgsz)
                self.infer_ms = (time.time() - t0) * 1000.0
                det = res.xyxy[0].detach().cpu().numpy()
                with self._lock:
                    self._out = det

        def stop(self):
            self._run = False
            self._evt.set()

    # ----- depth 샘플링 -----------------------------------------------------
    def _median_mm(patch, depth_scale):
        if patch is None or patch.size == 0:
            return None
        v = patch[patch > 0]
        if v.size == 0:
            return None
        return float(np.median(v)) * depth_scale * 1000.0

    def box_depth_mm(depth_al, x1, y1, x2, y2, depth_scale, shrink=0.6):
        """박스 내부를 shrink 비율로 축소한 영역의 median depth[mm]"""
        if depth_al is None:
            return None
        h, w = depth_al.shape[:2]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = (x2 - x1) * shrink / 2.0, (y2 - y1) * shrink / 2.0
        u0, u1 = int(max(0, cx - bw)), int(min(w, cx + bw))
        v0, v1 = int(max(0, cy - bh)), int(min(h, cy + bh))
        if u1 <= u0 or v1 <= v0:
            return None
        return _median_mm(depth_al[v0:v1, u0:u1], depth_scale)

    def sample_depth_mm(depth_al, u, v, depth_scale, radius=4):
        """중심 픽셀 주변 (2r+1)^2 창의 median depth[mm]"""
        if depth_al is None:
            return None
        h, w = depth_al.shape[:2]
        u, v = int(round(u)), int(round(v))
        u0, u1 = max(0, u - radius), min(w, u + radius + 1)
        v0, v1 = max(0, v - radius), min(h, v + radius + 1)
        if u1 <= u0 or v1 <= v0:
            return None
        return _median_mm(depth_al[v0:v1, u0:u1], depth_scale)

    # ----- 역투영 -----------------------------------------------------------
    def deproject_pixel(u, v, Z_mm, K=None, dist=None, factory_intrin=None):
        """픽셀 + depth -> color 카메라 좌표계 (X, Y, Z) [mm]"""
        if K is not None:
            if dist is not None:
                pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
                und = cv2.undistortPoints(pt, K, dist, P=K)
                u, v = float(und[0, 0, 0]), float(und[0, 0, 1])
            X = (u - K[0, 2]) / K[0, 0] * Z_mm
            Y = (v - K[1, 2]) / K[1, 1] * Z_mm
            return X, Y, float(Z_mm)
        p = rs.rs2_deproject_pixel_to_point(factory_intrin, [float(u), float(v)],
                                            float(Z_mm))
        return p[0], p[1], p[2]

    def color_to_depth_frame(X, Y, Z, R_c2d, t_c2d_mm):
        p = R_c2d @ np.array([X, Y, Z], dtype=np.float64) + t_c2d_mm
        return float(p[0]), float(p[1]), float(p[2])

    # ----- 시각화 -----------------------------------------------------------
    def draw_detection(img, x1, y1, x2, y2, uc, vc, label):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(img, (int(uc), int(vc)), 4, (0, 0, 255), -1)
        cv2.putText(img, label, (int(x1), max(14, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# ---------------------------------------------------------------------------
# ROS2 노드
# ---------------------------------------------------------------------------
class YoloDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # ---- 파라미터 -------------------------------------------------------
        self.declare_parameter('weights', os.path.join(SCRIPT_DIR, 'best.pt'))
        self.declare_parameter('params_xml', os.path.join(SCRIPT_DIR, 'camera_params.xml'))
        self.declare_parameter('target_class', '')      # '' 이면 모든 클래스
        self.declare_parameter('conf', 0.9)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('max_det', 20)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('every', 1)              # N 프레임마다 1회 추론
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('radius', 4)
        self.declare_parameter('shrink', 0.6)
        self.declare_parameter('show', True)            # OpenCV 창 표시
        # yolov5 저장소 경로 ('' 이면 torch.hub 캐시/홈 폴더에서 자동 탐색)
        self.declare_parameter('yolov5_dir', '')

        g = lambda n: self.get_parameter(n).value
        self.target_class = str(g('target_class')).strip()
        self.every = max(1, int(g('every')))
        self.radius = int(g('radius'))
        self.shrink = float(g('shrink'))
        self.show = bool(g('show'))

        # ---- YOLO 모델 ------------------------------------------------------
        self.get_logger().info(f'헬퍼 소스: {_HELPER_SRC}')
        wpath = os.path.abspath(os.path.expanduser(str(g('weights'))))
        self.get_logger().info(
            'YOLO 가중치: %s (존재: %s)' % (wpath, os.path.isfile(wpath)))
        self.model, self.names, self.device = load_yolo_model(
            wpath, conf=float(g('conf')), iou=float(g('iou')),
            max_det=int(g('max_det')),
            yolov5_dir=(str(g('yolov5_dir')).strip() or None))
        self.get_logger().info(f'모델 로드 완료 (device={self.device}, '
                               f'classes={self.names})')

        # ---- RealSense ------------------------------------------------------
        w, h, fps = int(g('width')), int(g('height')), int(g('fps'))
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        csp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        dsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.ci = csp.get_intrinsics()

        # ---- 카메라 내부 파라미터 (캘리브레이션 XML 우선) --------------------
        try:
            self.CAMK, self.CAMDIST = load_xml_camera_matrix_dist(g('params_xml'))
            self.get_logger().info(
                'XML intrinsics 사용: fx=%.2f fy=%.2f cx=%.2f cy=%.2f'
                % (self.CAMK[0, 0], self.CAMK[1, 1], self.CAMK[0, 2], self.CAMK[1, 2]))
        except Exception as e:
            self.CAMK = self.CAMDIST = None
            self.get_logger().warn(f'XML 로드 실패({e}) -> factory intrinsics 사용')

        self.R_c2d, self.t_c2d, ext_src = load_factory_extrinsic(csp, dsp)
        self.get_logger().info(f'color->depth extrinsic ({ext_src}), '
                               f'baseline={np.linalg.norm(self.t_c2d):.2f} mm')

        # ---- 발행자 ---------------------------------------------------------
        self.pub_point = self.create_publisher(Float64MultiArray, '/vision/object_camera', 10)
        self.pub_class = self.create_publisher(String, '/vision/object_class', 10)

        # ---- 추론 스레드 & 루프 타이머 --------------------------------------
        self.detector = DetectorThread(self.model, imgsz=int(g('imgsz')))
        self.detector.start()
        self.frame_i = 0
        self.create_timer(1.0 / fps, self.loop)
        self.get_logger().info('검출 시작 — /vision/object_camera 로 발행')

    # -----------------------------------------------------------------------
    def pick_best(self, det):
        """target_class 와 일치하는 검출 중 confidence 최대인 것 1개 선택"""
        best = None
        for x1, y1, x2, y2, conf, cls in det:
            name = self.names[int(cls)] if int(cls) < len(self.names) else str(int(cls))
            if self.target_class and name != self.target_class:
                continue
            if best is None or conf > best[4]:
                best = (x1, y1, x2, y2, conf, cls, name)
        return best

    def loop(self):
        frames = self.pipeline.wait_for_frames()
        color_f = frames.get_color_frame()
        if not color_f:
            return
        color = np.asanyarray(color_f.get_data())

        # N 프레임마다 1회만 추론 요청
        self.frame_i += 1
        if self.frame_i % self.every == 0:
            self.detector.submit(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

        det = self.detector.get()

        depth_al = None
        if det is not None and len(det):
            depth_f = self.align.process(frames).get_depth_frame()
            if depth_f:
                depth_al = np.asanyarray(depth_f.get_data())

        best = self.pick_best(det) if det is not None and len(det) else None

        if best is not None:
            x1, y1, x2, y2, conf, cls, name = best
            uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            Z = box_depth_mm(depth_al, x1, y1, x2, y2, self.depth_scale, self.shrink)
            if Z is None:
                Z = sample_depth_mm(depth_al, uc, vc, self.depth_scale, self.radius)

            if Z is not None:
                Xc, Yc, Zc = deproject_pixel(uc, vc, Z, K=self.CAMK,
                                             dist=self.CAMDIST, factory_intrin=self.ci)
                Xd, Yd, Zd = color_to_depth_frame(Xc, Yc, Zc, self.R_c2d, self.t_c2d)

                msg = Float64MultiArray()
                msg.data = [float(Xd), float(Yd), float(Zd),
                            float(conf), float(int(cls))]
                self.pub_point.publish(msg)
                self.pub_class.publish(String(data=name))

                label = '%s %.2f | X%.0f Y%.0f Z%.0f mm' % (name, conf, Xd, Yd, Zd)
            else:
                label = '%s %.2f | depth N/A' % (name, conf)

            if self.show:
                draw_detection(color, x1, y1, x2, y2, uc, vc, label)

        if self.show:
            cv2.putText(color, 'infer %.0f ms (%s)' % (self.detector.infer_ms, self.device),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('yolo_detector', color)
            cv2.waitKey(1)

    def shutdown(self):
        try:
            self.detector.stop()
            self.pipeline.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
