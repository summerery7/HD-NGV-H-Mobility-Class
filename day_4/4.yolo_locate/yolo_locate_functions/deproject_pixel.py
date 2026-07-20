#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def deproject_pixel(u, v, Z_mm, K=None, dist=None, factory_intrin=None):
    Zc = float(Z_mm)
    if K is not None:
        norm = cv2.undistortPoints(
            np.array([[[float(u), float(v)]]], dtype=np.float64), K, dist)
        xn, yn = float(norm[0, 0, 0]), float(norm[0, 0, 1])
        return xn * Zc, yn * Zc, Zc

    import pyrealsense2 as rs
    Xc, Yc, Zc = [c * 1000.0 for c in
                  rs.rs2_deproject_pixel_to_point(factory_intrin, [u, v], Zc / 1000.0)]
    return Xc, Yc, Zc
