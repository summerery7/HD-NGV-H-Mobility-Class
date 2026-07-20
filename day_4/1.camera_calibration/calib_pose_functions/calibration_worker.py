#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import cv2
import numpy as np
from PyQt5 import QtCore

from .find_corners import find_corners
from .make_object_points import make_object_points


class CalibrationWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    done_ok = QtCore.pyqtSignal(object)
    done_err = QtCore.pyqtSignal(str)

    def __init__(self, image_dir, cols, rows, square, parent=None):
        super().__init__(parent)
        self.image_dir = image_dir
        self.cols, self.rows, self.square = cols, rows, square

    def run(self):
        try:
            exts = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
            files = []
            for e in exts:
                files += glob.glob(os.path.join(self.image_dir, e))
            files = sorted(set(files))
            if not files:
                raise RuntimeError("이미지가 없습니다: %s" % self.image_dir)

            pattern = (self.cols, self.rows)
            objp = make_object_points(self.cols, self.rows, self.square)
            objpoints, imgpoints = [], []
            used, failed = 0, []
            image_size = None

            for f in files:
                img = cv2.imread(f)
                if img is None:
                    failed.append(os.path.basename(f))
                    continue
                if image_size is None:
                    image_size = (img.shape[1], img.shape[0])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = find_corners(gray, pattern, fast=False)
                if corners is not None:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    used += 1
                    self.progress.emit("  [O] %s" % os.path.basename(f))
                else:
                    failed.append(os.path.basename(f))
                    self.progress.emit("  [X] %s (코너 검출 실패)" % os.path.basename(f))

            if used < 3:
                raise RuntimeError("코너 검출 성공 이미지가 부족합니다 (%d장). "
                                   "다양한 각도로 최소 10장 이상 권장." % used)

            self.progress.emit("calibrateCamera 실행 중 ...")
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, image_size, None, None)
            newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)


            total_err = 0.0
            for i in range(len(objpoints)):
                proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)


                p1 = np.asarray(imgpoints[i], dtype=np.float32).reshape(-1, 1, 2)
                p2 = np.asarray(proj, dtype=np.float32).reshape(-1, 1, 2)
                err = cv2.norm(p1, p2, cv2.NORM_L2) / len(p2)
                total_err += err
            mean_err = total_err / len(objpoints)

            self.done_ok.emit(dict(
                rms=rms, mean_err=mean_err, K=K, dist=dist, newK=newK,
                roi=np.array(roi, dtype=int), image_size=image_size,
                used=used, total=len(files), failed=failed))
        except Exception as e:
            self.done_err.emit(str(e))
