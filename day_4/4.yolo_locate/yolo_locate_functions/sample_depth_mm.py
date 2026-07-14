#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sample_depth_mm(depth_raw, u, v, depth_scale, radius=4):
    if depth_raw is None:
        return None
    h, w = depth_raw.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (0 <= v < h and 0 <= u < w):
        return None
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
    patch = depth_raw[y0:y1, x0:x1].astype(np.float32)
    vals = patch[patch > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals)) * depth_scale * 1000.0
