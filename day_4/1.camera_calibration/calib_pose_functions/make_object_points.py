#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def make_object_points(cols, rows, square):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square)
    return objp
