#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def scrub_cv2_qt_plugin_path():
    for var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        if "cv2" in os.environ.get(var, ""):
            del os.environ[var]
