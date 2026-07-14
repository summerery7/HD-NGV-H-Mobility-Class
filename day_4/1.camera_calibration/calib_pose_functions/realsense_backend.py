#!/usr/bin/env python3
# -*- coding: utf-8 -*-


try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except Exception:
    rs = None
    HAS_REALSENSE = False


def realsense_device_connected():
    if not HAS_REALSENSE:
        return False
    try:
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False
