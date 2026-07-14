#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def box_depth_mm(depth_raw, x1, y1, x2, y2, depth_scale, shrink=0.6):
    if depth_raw is None:
        return None
    h, w = depth_raw.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * shrink, (y2 - y1) * shrink
    xa, xb = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    ya, yb = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    if xb <= xa or yb <= ya:
        return None
    region = depth_raw[ya:yb, xa:xb].astype(np.float32)
    vals = region[region > 0]
    if vals.size == 0:
        return None
    return float(np.median(vals)) * depth_scale * 1000.0
