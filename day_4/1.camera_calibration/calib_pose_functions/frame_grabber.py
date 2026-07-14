#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import QtCore


class FrameGrabber(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._running = True

    def run(self):
        while self._running:
            try:
                ok, color, depth = self.camera.read()
            except Exception as e:
                self.failed.emit(str(e))
                break
            if ok and color is not None:
                self.frame_ready.emit((color, depth))
            else:
                self.msleep(5)

    def stop(self):
        self._running = False
