#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import json
import datetime

import numpy as np
import cv2


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from calib_pose_functions import (
    SCRIPT_DIR,
    HAS_REALSENSE,
    realsense_device_connected,
    draw_axes,
    np_to_qpixmap,
    sample_depth_mm,
    compute_pose_dict,
    rotation_matrix_to_euler_zyx,
    CalibrationData,
    Camera,
    FrameGrabber,
    DetectorThread,
    CalibrationWorker,
)


from PyQt5 import QtCore, QtGui, QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense 캘리브레이션 & Pose 도구")
        self.resize(1180, 720)

        self.camera = None
        self.grabber = None
        self.detector = None
        self._detector_busy = False
        self._frame_seq = 0
        self.calib_worker = None
        self.calib = CalibrationData()

        self.last_color = None
        self.last_depth = None


        self._last_detect = 0.0
        self._detect_interval = 0.07
        self._detected_now = False
        self._detected_corners = None
        self._pose_corners = None
        self._pose_rvec = None
        self._pose_tvec = None
        self._pose_last = None
        self._last_readout = 0.0
        self._readout_interval = 0.4

        self._build_ui()

        default_xml = os.path.join(SCRIPT_DIR, "camera_params.xml")
        if os.path.exists(default_xml):
            try:
                self.calib.load_xml(default_xml)
                self.btn_save.setEnabled(True)
                self.status.showMessage("자동 로드: %s" % default_xml)
            except Exception:
                pass
        self._refresh_calib_labels()
        if not self.rs_ready:
            QtCore.QTimer.singleShot(0, self._notify_no_realsense)

    def _update_rs_label(self):
        if not HAS_REALSENSE:
            self.lbl_rs.setText("RealSense: 라이브러리 미설치")
            self.lbl_rs.setStyleSheet("color:#c62828; font-weight:bold;")
        elif self.rs_ready:
            self.lbl_rs.setText("RealSense: 사용 가능")
            self.lbl_rs.setStyleSheet("color:#2e7d32; font-weight:bold;")
        else:
            self.lbl_rs.setText("RealSense: 장치 없음")
            self.lbl_rs.setStyleSheet("color:#c62828; font-weight:bold;")

    def _refresh_realsense(self):
        self.rs_ready = realsense_device_connected()
        self._update_rs_label()
        self.status.showMessage("RealSense 재검색: %s" % ("연결됨" if self.rs_ready else "장치 없음"))

    def _notify_no_realsense(self):
        if not HAS_REALSENSE:
            msg = "pyrealsense2 라이브러리가 없습니다.\n설치 후 다시 실행하세요."
        else:
            msg = ("연결된 RealSense 장치가 없습니다.\n"
                   "카메라를 연결한 뒤 '장치 재검색'을 누르세요.")
        QtWidgets.QMessageBox.information(self, "RealSense 없음", msg)


    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)


        self.video = QtWidgets.QLabel("카메라 시작을 눌러주세요")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#1e1e1e; color:#aaa; border:1px solid #444;")
        root.addWidget(self.video, 3)


        side = QtWidgets.QVBoxLayout()
        root.addLayout(side, 2)


        cam_box = QtWidgets.QGroupBox("카메라 (RealSense 전용)")
        cam_l = QtWidgets.QFormLayout(cam_box)
        self.rs_ready = realsense_device_connected()
        self.lbl_rs = QtWidgets.QLabel()
        self.btn_rs_refresh = QtWidgets.QPushButton("장치 재검색")
        self.btn_rs_refresh.clicked.connect(self._refresh_realsense)
        self._update_rs_label()
        self.cmb_res = QtWidgets.QComboBox()
        for w, h in [(640, 480), (848, 480), (1280, 720), (960, 540), (1920, 1080)]:
            self.cmb_res.addItem("%dx%d" % (w, h), (w, h))
        self.btn_start = QtWidgets.QPushButton("카메라 시작")
        self.btn_stop = QtWidgets.QPushButton("정지"); self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_stop)
        rs_row = QtWidgets.QHBoxLayout()
        rs_row.addWidget(self.lbl_rs, 1)
        rs_row.addWidget(self.btn_rs_refresh)
        cam_l.addRow("상태", rs_row)
        cam_l.addRow("해상도", self.cmb_res)
        cam_l.addRow(hb)
        side.addWidget(cam_box)


        cb_box = QtWidgets.QGroupBox("체스보드 설정 (내부 코너 기준)")
        cb_l = QtWidgets.QFormLayout(cb_box)
        self.spin_cols = QtWidgets.QSpinBox(); self.spin_cols.setRange(2, 30); self.spin_cols.setValue(10)
        self.spin_rows = QtWidgets.QSpinBox(); self.spin_rows.setRange(2, 30); self.spin_rows.setValue(7)
        self.spin_square = QtWidgets.QDoubleSpinBox(); self.spin_square.setRange(0.1, 1000)
        self.spin_square.setValue(19.0); self.spin_square.setSuffix(" mm")
        cb_l.addRow("가로 코너 수 (cols)", self.spin_cols)
        cb_l.addRow("세로 코너 수 (rows)", self.spin_rows)
        cb_l.addRow("한 칸 크기", self.spin_square)
        side.addWidget(cb_box)


        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._tab_capture(), "1. 촬영/캡처")
        self.tabs.addTab(self._tab_calibrate(), "2. 캘리브레이션")
        self.tabs.addTab(self._tab_pose(), "3. Pose 추정")
        side.addWidget(self.tabs, 1)

        self.status = self.statusBar()
        if not HAS_REALSENSE:
            self.status.showMessage("준비됨. pyrealsense2 미설치 — RealSense 사용 불가")
        else:
            self.status.showMessage("준비됨. RealSense 장치: %s"
                                    % ("연결됨" if self.rs_ready else "없음"))

    def _tab_capture(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        self.lbl_savedir = QtWidgets.QLineEdit(os.path.join(SCRIPT_DIR, "chess"))
        btn_dir = QtWidgets.QPushButton("폴더 선택")
        btn_dir.clicked.connect(lambda: self._pick_dir(self.lbl_savedir))
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.lbl_savedir); hb.addWidget(btn_dir)
        l.addWidget(QtWidgets.QLabel("저장 폴더")); l.addLayout(hb)

        self.chk_preview = QtWidgets.QCheckBox("코너 미리보기(조준 보조)"); self.chk_preview.setChecked(True)
        self.chk_only_valid = QtWidgets.QCheckBox("코너 검출된 프레임만 저장"); self.chk_only_valid.setChecked(True)
        l.addWidget(self.chk_preview); l.addWidget(self.chk_only_valid)

        self.btn_capture = QtWidgets.QPushButton("현재 프레임 캡처  (단축키: S)")
        self.btn_capture.clicked.connect(self.capture_frame)
        self.btn_capture.setEnabled(False)
        l.addWidget(self.btn_capture)

        self.lbl_cap_status = QtWidgets.QLabel("검출: -")
        self.lbl_cap_count = QtWidgets.QLabel("저장된 이미지: 0장")
        l.addWidget(self.lbl_cap_status); l.addWidget(self.lbl_cap_count)
        l.addStretch(1)

        sc = QtWidgets.QShortcut(QtGui.QKeySequence("S"), self)
        sc.activated.connect(self.capture_frame)
        self._detected_now = False
        self._detected_corners = None
        self._update_count()
        return w

    def _tab_calibrate(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        self.lbl_imgdir = QtWidgets.QLineEdit(os.path.join(SCRIPT_DIR, "chess"))
        btn_dir = QtWidgets.QPushButton("폴더 선택")
        btn_dir.clicked.connect(lambda: self._pick_dir(self.lbl_imgdir))
        hb = QtWidgets.QHBoxLayout(); hb.addWidget(self.lbl_imgdir); hb.addWidget(btn_dir)
        l.addWidget(QtWidgets.QLabel("캘리브레이션 이미지 폴더")); l.addLayout(hb)

        self.btn_calib = QtWidgets.QPushButton("캘리브레이션 실행")
        self.btn_calib.clicked.connect(self.run_calibration)
        l.addWidget(self.btn_calib)

        hb2 = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("파라미터 저장(XML)")
        self.btn_load = QtWidgets.QPushButton("파라미터 로드(XML)")
        self.btn_save.clicked.connect(self.save_params)
        self.btn_load.clicked.connect(self.load_params)
        self.btn_save.setEnabled(False)
        hb2.addWidget(self.btn_save); hb2.addWidget(self.btn_load)
        l.addLayout(hb2)

        self.lbl_calib_info = QtWidgets.QLabel("캘리브레이션 결과 없음")
        self.lbl_calib_info.setWordWrap(True)
        self.lbl_calib_info.setStyleSheet("font-family:monospace;")
        l.addWidget(self.lbl_calib_info)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        l.addWidget(self.log, 1)
        return w

    def _tab_pose(self):
        w = QtWidgets.QWidget(); l = QtWidgets.QVBoxLayout(w)
        self.chk_undistort = QtWidgets.QCheckBox("왜곡 보정 미리보기 (축은 원본에 표시)")
        self.chk_flip = QtWidgets.QCheckBox("코너 순서 뒤집기(원점 반대편으로)")
        l.addWidget(self.chk_undistort); l.addWidget(self.chk_flip)

        opt = QtWidgets.QFormLayout()

        opt.addRow("solvePnP 내부행렬", QtWidgets.QLabel("camera_matrix"))
        self.spin_readout = QtWidgets.QDoubleSpinBox()
        self.spin_readout.setRange(0.05, 2.0); self.spin_readout.setSingleStep(0.1)
        self.spin_readout.setValue(0.4); self.spin_readout.setSuffix(" s")
        self.spin_readout.valueChanged.connect(
            lambda v: setattr(self, "_readout_interval", float(v)))
        opt.addRow("값 갱신 주기", self.spin_readout)
        l.addLayout(opt)


        note = QtWidgets.QLabel("이동 X/Y/Z 와 회전은 camera_matrix 로 구한 자세에 "
                                "공장 extrinsic(color→depth)을 합성한 depth(IR) 프레임 기준 값입니다.")
        note.setStyleSheet("color:#2e7d32;")
        note.setWordWrap(True)
        l.addWidget(note)


        self.lbl_pose_status = QtWidgets.QLabel("검출 상태: -")
        l.addWidget(self.lbl_pose_status)


        grid_box = QtWidgets.QGroupBox("체스보드 자세 (카메라 좌표계 기준)")
        grid = QtWidgets.QGridLayout(grid_box)
        self.pose_value_labels = {}
        rows = [
            ("roll",  "회전 X (roll)",  "deg"),
            ("pitch", "회전 Y (pitch)", "deg"),
            ("yaw",   "회전 Z (yaw)",   "deg"),
            ("tx",    "이동 X",          "mm"),
            ("ty",    "이동 Y",          "mm"),
            ("tz",    "이동 Z",          "mm"),
            ("line",  "직선거리",        "mm"),
        ]
        for i, (key, title, unit) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(title + " :"), i, 0)
            val = QtWidgets.QLabel("---")
            val.setStyleSheet("font-family:monospace; font-size:14px; font-weight:bold;")
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(val, i, 1)
            grid.addWidget(QtWidgets.QLabel(unit), i, 2)
            self.pose_value_labels[key] = val
        grid.setColumnStretch(1, 1)
        l.addWidget(grid_box)


        save_box = QtWidgets.QGroupBox("pose 저장")
        sv = QtWidgets.QVBoxLayout(save_box)
        self.btn_pose_json = QtWidgets.QPushButton("현재 pose 저장 (JSON)")
        self.btn_pose_json.clicked.connect(self.save_pose_json)
        sv.addWidget(self.btn_pose_json)
        csv_row = QtWidgets.QHBoxLayout()
        self.lbl_csv = QtWidgets.QLineEdit(os.path.join(SCRIPT_DIR, "pose"))
        btn_csv_dir = QtWidgets.QPushButton("...")
        btn_csv_dir.setFixedWidth(32)
        btn_csv_dir.clicked.connect(self._pick_csv)
        csv_row.addWidget(self.lbl_csv, 1); csv_row.addWidget(btn_csv_dir)
        sv.addWidget(QtWidgets.QLabel("CSV 파일"))
        sv.addLayout(csv_row)
        self.btn_pose_csv = QtWidgets.QPushButton("현재 pose를 CSV에 한 줄 추가")
        self.btn_pose_csv.clicked.connect(self.append_pose_csv)
        sv.addWidget(self.btn_pose_csv)
        l.addWidget(save_box)

        self.lbl_pose_hint = QtWidgets.QLabel("캘리브레이션 후 사용 가능합니다.")
        self.lbl_pose_hint.setStyleSheet("color:#888;")
        l.addWidget(self.lbl_pose_hint)
        l.addStretch(1)
        return w


    def start_camera(self):
        if not realsense_device_connected():
            self.rs_ready = False
            self._update_rs_label()
            QtWidgets.QMessageBox.warning(
                self, "RealSense 없음",
                "연결된 RealSense 장치가 없습니다.\n카메라를 연결한 뒤 '장치 재검색'을 누르세요.")
            return
        self.rs_ready = True
        self._update_rs_label()
        w, h = self.cmb_res.currentData()
        try:
            self.camera = Camera(source="realsense", width=w, height=h)
            self.camera.open()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "카메라 오류", str(e))
            self.camera = None
            return

        self._detector_busy = False
        self._detected_now = False
        self._detected_corners = None
        self._pose_corners = None
        self._pose_rvec = None
        self._pose_tvec = None


        self.detector = DetectorThread()
        self.detector.result_ready.connect(self.on_detection)
        self.detector.start()

        self.grabber = FrameGrabber(self.camera)
        self.grabber.frame_ready.connect(self.on_frame)
        self.grabber.failed.connect(self.on_grabber_failed)
        self.grabber.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        depth_txt = " + depth" if (self.camera.has_depth) else ""
        self.status.showMessage("촬영 중: RealSense %dx%d%s" % (w, h, depth_txt))

    def stop_camera(self):

        if self.grabber:
            self.grabber.stop(); self.grabber.wait(2000); self.grabber = None
        if self.detector:
            self.detector.stop(); self.detector.wait(2000); self.detector = None
        self._detector_busy = False
        if self.camera:
            self.camera.close(); self.camera = None
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.status.showMessage("정지됨")

    def on_grabber_failed(self, msg):
        self.status.showMessage("프레임 취득 실패: %s" % msg)
        self.stop_camera()


    def pattern(self):
        return (self.spin_cols.value(), self.spin_rows.value())

    def _detect_scale(self):
        if self.last_color is None:
            return 1.0
        w = self.last_color.shape[1]
        return 1.0 if w <= 700 else 640.0 / w

    def _detect_due(self):
        now = time.monotonic()
        if now - self._last_detect >= self._detect_interval:
            self._last_detect = now
            return True
        return False

    def on_frame(self, data):
        color, depth = data
        self.last_color = color
        self.last_depth = depth
        disp = color.copy()
        idx = self.tabs.currentIndex()

        if idx == 0:
            disp = self._overlay_capture(disp)
        elif idx == 2:
            disp = self._overlay_pose(color, disp)


        if (self.detector is not None and not self._detector_busy
                and self._detect_due()):
            job = self._build_detect_job(idx, color, depth)
            if job is not None:
                self._detector_busy = True
                self.detector.submit(job)


        self.video.setPixmap(np_to_qpixmap(disp).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


    def _build_detect_job(self, idx, color, depth):
        self._frame_seq += 1
        seq = self._frame_seq
        if idx == 0:
            if not self.chk_preview.isChecked():
                return None
            return {"mode": "capture", "seq": seq, "color": color,
                    "pattern": self.pattern(), "scale": self._detect_scale()}
        if idx == 2:
            if not self.calib.valid():
                return None
            return {"mode": "pose", "seq": seq, "color": color, "depth": depth,
                    "pattern": self.pattern(), "scale": self._detect_scale(),
                    "cols": self.spin_cols.value(), "rows": self.spin_rows.value(),
                    "square": float(self.spin_square.value()),
                    "flip": self.chk_flip.isChecked(),
                    "K": self.calib.K, "dist": self.calib.dist,


                    "K_pose": self.calib.K,
                    "depth_scale": (self.camera.depth_scale if self.camera else 1.0),
                    "R_c2d": (self.camera.R_c2d if self.camera else None),
                    "t_c2d_mm": (self.camera.t_c2d_mm if self.camera else None)}
        return None


    def on_detection(self, result):
        self._detector_busy = False
        if self.detector is None:
            return
        if "error" in result:
            return
        mode = result.get("mode")
        if mode == "capture":
            self._detected_corners = result["corners"]
            self._detected_now = result["corners"] is not None
            self._render_capture_status()
        elif mode == "pose":
            corners = result["corners"]
            self._pose_corners = corners
            if not result.get("ok"):

                self._pose_rvec = self._pose_tvec = None
                self._set_pose_status(False)
            else:
                self._pose_rvec = result["rvec"]
                self._pose_tvec = result["tvec"]
                self._pose_last = result["pose"]

                if time.monotonic() - self._last_readout >= self._readout_interval:
                    self._last_readout = time.monotonic()
                    self._render_pose_values()
                self._set_pose_status(True)


    def _overlay_capture(self, disp):
        if not self.chk_preview.isChecked():
            self.lbl_cap_status.setText("검출: (미리보기 꺼짐)")
            self.lbl_cap_status.setStyleSheet("color:#888;")
            return disp
        if self._detected_now and self._detected_corners is not None:
            cv2.drawChessboardCorners(disp, self.pattern(), self._detected_corners, True)
        self._render_capture_status()
        return disp

    def _render_capture_status(self):
        if not self.chk_preview.isChecked():
            self.lbl_cap_status.setText("검출: (미리보기 꺼짐)")
            self.lbl_cap_status.setStyleSheet("color:#888;")
        elif self._detected_now and self._detected_corners is not None:
            self.lbl_cap_status.setText("검출: O  (S 키로 캡처)")
            self.lbl_cap_status.setStyleSheet("color:#2e7d32; font-weight:bold;")
        else:
            self.lbl_cap_status.setText("검출: X")
            self.lbl_cap_status.setStyleSheet("color:#c62828;")

    def _overlay_pose(self, color, disp):

        if not self.calib.valid():
            cv2.putText(disp, "Calibration required (camera_matrix)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self._set_pose_status(None)
            return disp
        K = self.calib.K
        dist = self.calib.dist


        if self.chk_undistort.isChecked():
            newK = self.calib.newK if self.calib.newK is not None else K
            return cv2.undistort(color, K, dist, None, newK)


        if self._pose_corners is not None:
            cv2.drawChessboardCorners(disp, self.pattern(), self._pose_corners, True)
        if self._pose_rvec is not None:
            draw_axes(disp, K, dist, self._pose_rvec, self._pose_tvec,
                      self.spin_square.value() * 3)
        return disp

    def _set_pose_status(self, detected):
        if detected is None:
            self.lbl_pose_status.setText("검출 상태: 캘리브레이션 필요")
            self.lbl_pose_status.setStyleSheet("color:#c62828;")
        elif detected:
            self.lbl_pose_status.setText("검출 상태: 검출됨 (실시간 갱신)")
            self.lbl_pose_status.setStyleSheet("color:#2e7d32; font-weight:bold;")
        else:
            self.lbl_pose_status.setText("검출 상태: 검출 안됨 (최근 값 유지)")
            self.lbl_pose_status.setStyleSheet("color:#e65100; font-weight:bold;")

    def _sample_depth_mm(self, depth, u, v, radius=3):
        ds = self.camera.depth_scale if self.camera else 1.0
        return sample_depth_mm(depth, u, v, ds, radius)

    def _store_pose(self, rvec, tvec, pts, depth, K_pose, dist):
        ds = self.camera.depth_scale if self.camera else 1.0
        R_c2d = self.camera.R_c2d if self.camera else None
        t_c2d = self.camera.t_c2d_mm if self.camera else None
        self._pose_last = compute_pose_dict(
            rvec, tvec, pts, depth, K_pose, dist, ds,
            self.spin_cols.value(), self.spin_rows.value(),
            float(self.spin_square.value()), R_c2d, t_c2d)

    def _store_pose_legacy(self, rvec, tvec, pts, depth, K_pose, dist):
        R, _ = cv2.Rodrigues(rvec)
        roll, pitch, yaw = rotation_matrix_to_euler_zyx(R)
        tx0, ty0, tz0 = [float(x) for x in tvec.flatten()]
        u, v = pts.reshape(-1, 2)[0]

        d_mm = self._sample_depth_mm(depth, int(round(u)), int(round(v)))


        if d_mm is not None:
            norm = cv2.undistortPoints(
                np.array([[[float(u), float(v)]]], dtype=np.float64), K_pose, dist)
            xn, yn = float(norm[0, 0, 0]), float(norm[0, 0, 1])
            tz = float(d_mm)
            tx = xn * tz
            ty = yn * tz
        else:

            tx, ty, tz = tx0, ty0, tz0


        line = float(np.sqrt(tx * tx + ty * ty + tz * tz))

        self._pose_last = dict(
            roll=roll, pitch=pitch, yaw=yaw, tx=tx, ty=ty, tz=tz, line=line,
            rvec=[float(x) for x in rvec.flatten()],
            tvec=[tx, ty, tz],
            R=[[float(x) for x in row] for row in R],
            K_pose=[[float(x) for x in row] for row in K_pose],
            cols=self.spin_cols.value(), rows=self.spin_rows.value(),
            square=float(self.spin_square.value()),
            time=datetime.datetime.now().isoformat(timespec="seconds"))

    def _render_pose_values(self):
        v = self._pose_last
        if not v:
            return
        fmt = {"roll": "%.3f", "pitch": "%.3f", "yaw": "%.3f",
               "tx": "%.2f", "ty": "%.2f", "tz": "%.2f", "line": "%.2f"}
        for key, lbl in self.pose_value_labels.items():
            val = v.get(key)
            lbl.setText("---" if val is None else fmt[key] % val)


    def _pick_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "CSV 파일 선택", self.lbl_csv.text(), "CSV (*.csv)")
        if path:
            self.lbl_csv.setText(path)

    def save_pose_json(self):
        if not self._pose_last:
            QtWidgets.QMessageBox.information(self, "저장", "아직 저장할 pose 가 없습니다.")
            return
        default = os.path.join(
            SCRIPT_DIR, "pose_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "pose 저장(JSON)", default, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._pose_last, f, indent=2, ensure_ascii=False)
            self.status.showMessage("pose 저장됨: %s" % path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "저장 실패", str(e))

    def append_pose_csv(self):
        if not self._pose_last:
            QtWidgets.QMessageBox.information(self, "저장", "아직 저장할 pose 가 없습니다.")
            return
        path = self.lbl_csv.text().strip()
        if not path:
            return
        v = self._pose_last
        cols = ["time", "roll", "pitch", "yaw", "tx", "ty", "tz", "line"]
        try:
            new_file = not os.path.exists(path)
            with open(path, "a", encoding="utf-8") as f:
                if new_file:
                    f.write(",".join(cols) + "\n")
                row = []
                for c in cols:
                    val = v.get(c)
                    row.append("" if val is None else (val if c == "time" else "%.4f" % val))
                f.write(",".join(str(x) for x in row) + "\n")
            self.status.showMessage("CSV에 추가됨: %s" % path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "저장 실패", str(e))


    def capture_frame(self):
        if self.last_color is None:
            return
        if self.chk_only_valid.isChecked() and not self._detected_now:
            self.status.showMessage("코너 미검출 프레임 — 저장하지 않음")
            return
        save_dir = self.lbl_savedir.text().strip()
        os.makedirs(save_dir, exist_ok=True)
        fname = self._next_filename(save_dir)
        cv2.imwrite(fname, self.last_color)
        self.status.showMessage("저장: %s" % fname)
        self._update_count()

    def _next_filename(self, directory, prefix="chess", ext="png"):
        nums = []
        for f in os.listdir(directory):
            if f.startswith(prefix + " (") and f.endswith(").%s" % ext):
                try:
                    nums.append(int(f[len(prefix) + 2:-(len(ext) + 2)]))
                except ValueError:
                    pass
        n = (max(nums) + 1) if nums else 0
        return os.path.join(directory, "%s (%d).%s" % (prefix, n, ext))

    def _update_count(self):
        d = self.lbl_savedir.text().strip()
        cnt = 0
        if os.path.isdir(d):
            for e in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
                cnt += len(glob.glob(os.path.join(d, e)))
        self.lbl_cap_count.setText("저장된 이미지: %d장" % cnt)


    def run_calibration(self):
        img_dir = self.lbl_imgdir.text().strip()
        if not os.path.isdir(img_dir):
            QtWidgets.QMessageBox.warning(self, "경고", "이미지 폴더가 없습니다.")
            return
        self.log.clear()
        self.log.appendPlainText("캘리브레이션 시작: %s" % img_dir)
        self.btn_calib.setEnabled(False)
        self.calib_worker = CalibrationWorker(
            img_dir, self.spin_cols.value(), self.spin_rows.value(), self.spin_square.value())
        self.calib_worker.progress.connect(self.log.appendPlainText)
        self.calib_worker.done_ok.connect(self.on_calib_ok)
        self.calib_worker.done_err.connect(self.on_calib_err)
        self.calib_worker.start()

    def on_calib_ok(self, r):
        self.calib.from_result(r)
        self.btn_calib.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.log.appendPlainText("완료: 사용 %d/%d장, RMS=%.4f, 평균재투영오차=%.4f px"
                                 % (r["used"], r["total"], r["rms"], r["mean_err"]))
        self._refresh_calib_labels()
        self.status.showMessage("캘리브레이션 완료 (RMS=%.4f)" % r["rms"])

    def on_calib_err(self, msg):
        self.btn_calib.setEnabled(True)
        self.log.appendPlainText("오류: %s" % msg)
        QtWidgets.QMessageBox.critical(self, "캘리브레이션 실패", msg)

    def _refresh_calib_labels(self):
        if self.calib.valid():
            K = self.calib.K
            info = ("K = [[%.2f, 0, %.2f],\n     [0, %.2f, %.2f],\n     [0, 0, 1]]\n"
                    "dist = %s\n크기 = %s"
                    % (K[0, 0], K[0, 2], K[1, 1], K[1, 2],
                       np.array2string(self.calib.dist.flatten(), precision=5),
                       str(self.calib.image_size)))
            self.lbl_calib_info.setText(info)
            self.lbl_pose_hint.setText("체스보드를 비추면 pose 가 표시됩니다.")
        else:
            self.lbl_calib_info.setText("캘리브레이션 결과 없음")

    def save_params(self):
        if not self.calib.valid():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "파라미터 저장", os.path.join(SCRIPT_DIR, "camera_params.xml"), "XML (*.xml)")
        if path:
            self.calib.save_xml(path)
            self.status.showMessage("저장됨: %s" % path)

    def load_params(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "파라미터 로드", SCRIPT_DIR, "XML (*.xml)")
        if not path:
            return
        try:
            self.calib.load_xml(path)
            self.btn_save.setEnabled(True)
            self._refresh_calib_labels()
            self.status.showMessage("로드됨: %s" % path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "로드 실패", str(e))


    def _pick_dir(self, line_edit):

            start_path = os.path.abspath(line_edit.text())

            d = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 선택", start_path)
            if d:

                line_edit.setText(d)
                self._update_count()
    def closeEvent(self, e):
        self.stop_camera()
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
