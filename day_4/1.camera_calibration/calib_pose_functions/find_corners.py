#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

from .config import SUBPIX_CRITERIA


def find_corners(gray, pattern, fast=False):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ret:
        return None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners
