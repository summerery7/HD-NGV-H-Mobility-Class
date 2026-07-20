#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime

import cv2
import numpy as np

from .rotation_matrix_to_euler_zyx import rotation_matrix_to_euler_zyx
from .sample_depth_mm import sample_depth_mm


def compute_pose_dict(rvec, tvec, pts, depth, K_pose, dist, depth_scale,
                      cols, rows, square, R_c2d=None, t_c2d_mm=None):

    R_cb, _ = cv2.Rodrigues(rvec)
    t_cb = np.array([float(x) for x in tvec.flatten()], dtype=np.float64)


    if R_c2d is not None and t_c2d_mm is not None:
        R_c2d = np.asarray(R_c2d, dtype=np.float64)
        t_c2d_mm = np.asarray(t_c2d_mm, dtype=np.float64)
        R_db = R_c2d @ R_cb
        t_db = R_c2d @ t_cb + t_c2d_mm
    else:
        R_db, t_db = R_cb, t_cb

    roll, pitch, yaw = rotation_matrix_to_euler_zyx(R_db)
    tx, ty, tz = [float(x) for x in t_db]
    line = float(np.sqrt(tx * tx + ty * ty + tz * tz))


    u, v = pts.reshape(-1, 2)[0]
    d_mm = sample_depth_mm(depth, int(round(u)), int(round(v)), depth_scale)

    return dict(
        roll=roll, pitch=pitch, yaw=yaw, tx=tx, ty=ty, tz=tz, line=line,
        rvec=[float(x) for x in rvec.flatten()],
        tvec=[tx, ty, tz],
        R=[[float(x) for x in row] for row in R_db],
        R_color=[[float(x) for x in row] for row in R_cb],
        tvec_color=[float(x) for x in t_cb],
        realdepth_mm=(None if d_mm is None else float(d_mm)),
        K_pose=[[float(x) for x in row] for row in K_pose],
        cols=cols, rows=rows, square=float(square),
        time=datetime.datetime.now().isoformat(timespec="seconds"))
