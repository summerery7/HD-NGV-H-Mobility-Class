#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def color_to_depth_frame(Xc, Yc, Zc, R_c2d, t_c2d_mm):
    Xd, Yd, Zd = R_c2d @ np.array([Xc, Yc, Zc], dtype=np.float64) + t_c2d_mm
    return float(Xd), float(Yd), float(Zd)
