#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PyQt5 import QtGui


def np_to_qpixmap(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())
