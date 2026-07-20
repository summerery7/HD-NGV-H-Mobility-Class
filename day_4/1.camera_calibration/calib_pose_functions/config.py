#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import cv2

IS_WINDOWS = sys.platform.startswith("win")
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(_PKG_DIR)


RGB_TO_DEPTH_X_MM = 15.0
