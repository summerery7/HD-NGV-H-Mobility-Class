#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def draw_axes(img, K, dist, rvec, tvec, length):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, length, 3)
        return
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    pts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    pts = pts.reshape(-1, 2).astype(int)
    o = tuple(pts[0])
    cv2.line(img, o, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(img, o, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(img, o, tuple(pts[3]), (255, 0, 0), 3)
