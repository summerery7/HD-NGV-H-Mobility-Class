#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from PyQt5 import QtCore

from .find_corners_fast import find_corners_fast
from .make_object_points import make_object_points
from .compute_pose_dict import compute_pose_dict


class DetectorThread(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QtCore.QMutex()
        self._cond = QtCore.QWaitCondition()
        self._job = None
        self._running = True

    def submit(self, job):
        self._mutex.lock()
        self._job = job
        self._cond.wakeOne()
        self._mutex.unlock()

    def run(self):
        while True:
            self._mutex.lock()
            while self._running and self._job is None:
                self._cond.wait(self._mutex)
            if not self._running:
                self._mutex.unlock()
                return
            job = self._job
            self._job = None
            self._mutex.unlock()

            try:
                result = self._process(job)
            except Exception as e:
                result = {"mode": job.get("mode"), "seq": job.get("seq"),
                          "error": str(e)}
            self.result_ready.emit(result)

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._cond.wakeOne()
        self._mutex.unlock()


    def _process(self, job):
        mode = job["mode"]
        gray = cv2.cvtColor(job["color"], cv2.COLOR_BGR2GRAY)
        corners = find_corners_fast(gray, job["pattern"], job["scale"])

        if mode == "capture":
            return {"mode": "capture", "seq": job["seq"], "corners": corners}


        if corners is None:
            return {"mode": "pose", "seq": job["seq"], "corners": None, "ok": False}
        pts = corners[::-1] if job["flip"] else corners
        objp = make_object_points(job["cols"], job["rows"], job["square"])
        ok, rvec, tvec = cv2.solvePnP(objp, pts, job["K_pose"], job["dist"])
        if not ok:
            return {"mode": "pose", "seq": job["seq"], "corners": corners, "ok": False}
        pose = compute_pose_dict(rvec, tvec, pts, job["depth"], job["K_pose"],
                                 job["dist"], job["depth_scale"],
                                 job["cols"], job["rows"], job["square"],
                                 job.get("R_c2d"), job.get("t_c2d_mm"))
        return {"mode": "pose", "seq": job["seq"], "corners": corners,
                "ok": True, "rvec": rvec, "tvec": tvec, "pose": pose}
