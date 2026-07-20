#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from .config import IS_WINDOWS
from .realsense_backend import rs, HAS_REALSENSE


class Camera:
    def __init__(self, source="realsense", width=640, height=480, fps=30, cam_index=0):
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.cam_index = cam_index
        self.pipeline = None
        self.align = None
        self.cap = None
        self.depth_scale = 1.0
        self.has_depth = False

        self.K_factory = None
        self.dist_factory = None
        self.t_d2c_mm = None
        self.baseline_mm = 0.0

        self.R_c2d = None
        self.t_c2d_mm = None

    def open(self):
        if self.source == "realsense":
            if not HAS_REALSENSE:
                raise RuntimeError("pyrealsense2 가 설치되어 있지 않습니다. 웹캠 모드를 사용하세요.")
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            try:
                cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                self.has_depth = True
            except Exception:
                self.has_depth = False
            profile = self.pipeline.start(cfg)

            try:
                csp = profile.get_stream(rs.stream.color).as_video_stream_profile()
                ci = csp.get_intrinsics()
                self.K_factory = np.array([[ci.fx, 0, ci.ppx],
                                           [0, ci.fy, ci.ppy],
                                           [0, 0, 1]], dtype=np.float64)
                cc = list(ci.coeffs) + [0.0] * 5
                self.dist_factory = np.array(cc[:5], dtype=np.float64).reshape(1, 5)
            except Exception:
                self.K_factory = self.dist_factory = None
            if self.has_depth:
                try:
                    ds = profile.get_device().first_depth_sensor()
                    self.depth_scale = ds.get_depth_scale()
                    self.align = rs.align(rs.stream.color)
                    dsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()

                    ex = dsp.get_extrinsics_to(csp)
                    t = np.array(ex.translation, dtype=np.float64)
                    self.t_d2c_mm = t * 1000.0
                    self.baseline_mm = float(np.linalg.norm(t) * 1000.0)


                    ex_c2d = csp.get_extrinsics_to(dsp)
                    self.R_c2d = np.asarray(ex_c2d.rotation,
                                            dtype=np.float64).reshape(3, 3).T
                    self.t_c2d_mm = np.asarray(ex_c2d.translation,
                                               dtype=np.float64) * 1000.0
                except Exception:
                    self.has_depth = False
        else:
            if IS_WINDOWS:
                self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.cam_index)
            if not self.cap.isOpened():
                raise RuntimeError("웹캠을 열 수 없습니다 (index=%d)." % self.cam_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.source == "realsense":
            frames = self.pipeline.wait_for_frames(3000)
            if self.align is not None:
                frames = self.align.process(frames)
            color = frames.get_color_frame()
            if not color:
                return False, None, None


            color_img = np.array(color.get_data())
            depth_raw = None
            if self.has_depth:
                d = frames.get_depth_frame()
                if d:
                    depth_raw = np.array(d.get_data())
            return True, color_img, depth_raw
        else:
            ok, img = self.cap.read()
            if not ok or img is None:
                return False, None, None
            return True, img, None

    def close(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
