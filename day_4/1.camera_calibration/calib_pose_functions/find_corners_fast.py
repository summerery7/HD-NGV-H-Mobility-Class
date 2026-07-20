#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from .config import SUBPIX_CRITERIA


def find_corners_fast(gray, pattern, downscale=1.0):
    if downscale < 0.999:
        small = cv2.resize(gray, None, fx=downscale, fy=downscale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ret:
        return None
    if downscale < 0.999:
        corners = (corners / downscale).astype(np.float32)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), SUBPIX_CRITERIA)
    return corners
