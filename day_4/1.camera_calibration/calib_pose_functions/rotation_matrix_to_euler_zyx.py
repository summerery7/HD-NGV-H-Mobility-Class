#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def rotation_matrix_to_euler_zyx(R):
    r11, r21, r31 = R[0, 0], R[1, 0], R[2, 0]
    r32, r33 = R[2, 1], R[2, 2]
    cos_pitch = np.sqrt(r11 ** 2 + r21 ** 2)
    if cos_pitch > 1e-6:
        roll = np.arctan2(r32, r33)
        pitch = np.arctan2(-r31, cos_pitch)
        yaw = np.arctan2(r21, r11)
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-r31, cos_pitch)
        yaw = 0.0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
