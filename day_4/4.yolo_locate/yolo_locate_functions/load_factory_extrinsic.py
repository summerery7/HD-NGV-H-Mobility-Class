#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import R_C2D_FALLBACK, T_C2D_MM_FALLBACK


def load_factory_extrinsic(color_profile, depth_profile):
    try:
        ext = color_profile.get_extrinsics_to(depth_profile)
        R_c2d = np.asarray(ext.rotation, dtype=np.float64).reshape(3, 3).T
        t_c2d_mm = np.asarray(ext.translation, dtype=np.float64) * 1000.0
        return R_c2d, t_c2d_mm, "SDK 실측"
    except Exception:
        return R_C2D_FALLBACK, T_C2D_MM_FALLBACK, "하드코딩 fallback"
