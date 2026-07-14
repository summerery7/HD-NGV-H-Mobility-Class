#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


R_C2D_FALLBACK = np.array([
    [0.99997,  -0.006737, -0.003796],
    [0.006748,  0.999973,  0.002933],
    [0.003776, -0.002958,  0.999988],
], dtype=np.float64)

T_C2D_MM_FALLBACK = np.array([-14.760, 0.071, -0.395], dtype=np.float64)
