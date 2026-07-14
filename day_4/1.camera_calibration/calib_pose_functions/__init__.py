#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .qt_env import scrub_cv2_qt_plugin_path

scrub_cv2_qt_plugin_path()


from .config import (
    IS_WINDOWS,
    SUBPIX_CRITERIA,
    SCRIPT_DIR,
    RGB_TO_DEPTH_X_MM,
)
from .realsense_backend import rs, HAS_REALSENSE, realsense_device_connected

from .make_object_points import make_object_points
from .find_corners import find_corners
from .find_corners_fast import find_corners_fast
from .rotation_matrix_to_euler_zyx import rotation_matrix_to_euler_zyx
from .draw_axes import draw_axes
from .np_to_qpixmap import np_to_qpixmap
from .sample_depth_mm import sample_depth_mm
from .compute_pose_dict import compute_pose_dict

from .calibration_data import CalibrationData
from .camera import Camera
from .frame_grabber import FrameGrabber
from .detector_thread import DetectorThread
from .calibration_worker import CalibrationWorker

__all__ = [

    "scrub_cv2_qt_plugin_path",
    "IS_WINDOWS",
    "SUBPIX_CRITERIA",
    "SCRIPT_DIR",
    "RGB_TO_DEPTH_X_MM",
    "rs",
    "HAS_REALSENSE",
    "realsense_device_connected",

    "make_object_points",
    "find_corners",
    "find_corners_fast",
    "rotation_matrix_to_euler_zyx",
    "draw_axes",
    "np_to_qpixmap",
    "sample_depth_mm",
    "compute_pose_dict",

    "CalibrationData",
    "Camera",
    "FrameGrabber",
    "DetectorThread",
    "CalibrationWorker",
]
