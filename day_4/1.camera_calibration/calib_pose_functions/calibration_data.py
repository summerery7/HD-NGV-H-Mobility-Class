#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

import numpy as np


class CalibrationData:
    def __init__(self):
        self.image_size = None
        self.K = None
        self.newK = None
        self.dist = None
        self.roi = None

    def valid(self):
        return self.K is not None and self.dist is not None

    def from_result(self, r):
        self.image_size = r["image_size"]
        self.K = r["K"]
        self.newK = r["newK"]
        self.dist = r["dist"]
        self.roi = r["roi"]


    def save_xml(self, path):
        root = ET.Element("root")
        about = ET.SubElement(root, "about")
        about.set("author", "calib_pose_tool")

        def write_mat(tag, arr, as_int=False):
            node = ET.SubElement(root, tag)
            for i, v in enumerate(np.array(arr).flatten()):
                c = ET.SubElement(node, "data%d" % i)
                c.text = str(int(v)) if as_int else str(float(v))

        write_mat("camera_matrix", self.K)
        write_mat("new_camera_matrix", self.newK if self.newK is not None else self.K)
        write_mat("camera_distortion", self.dist)
        if self.roi is not None:
            write_mat("roi", self.roi, as_int=True)
        if self.image_size is not None:
            write_mat("image_size", np.array(self.image_size), as_int=True)
        ET.ElementTree(root).write(path, encoding="UTF-8")


    def load_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        def read_mat(tag, n):
            node = root.find(tag)
            if node is None:
                return None
            d = {c.tag: c.text for c in list(node)}
            try:
                return np.array([float(d["data%d" % i]) for i in range(n)])
            except (KeyError, TypeError, ValueError):
                return None

        m = read_mat("camera_matrix", 9)
        if m is not None:
            self.K = m.reshape(3, 3)
        nm = read_mat("new_camera_matrix", 9)
        if nm is not None:
            self.newK = nm.reshape(3, 3)
        elif self.K is not None:
            self.newK = self.K.copy()
        dd = read_mat("camera_distortion", 5)
        if dd is not None:
            self.dist = dd.reshape(1, 5)
        rr = read_mat("roi", 4)
        if rr is not None:
            self.roi = rr.astype(int)
        isz = read_mat("image_size", 2)
        if isz is not None:
            self.image_size = (int(isz[0]), int(isz[1]))
        if not self.valid():
            raise ValueError("camera_matrix / camera_distortion 를 읽지 못했습니다.")
