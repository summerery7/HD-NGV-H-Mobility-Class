#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

import numpy as np


def load_xml_camera_matrix_dist(param_file):
    root = ET.parse(param_file).getroot()

    def read(tag, n):
        node = root.find(tag)
        if node is None:
            raise ValueError("%s 에 <%s> 노드가 없습니다." % (param_file, tag))
        d = {c.tag: c.text for c in list(node)}
        return np.array([float(d["data%d" % i]) for i in range(n)])

    K = read("camera_matrix", 9).reshape(3, 3)
    dist = read("camera_distortion", 5).reshape(1, 5)
    return K, dist
