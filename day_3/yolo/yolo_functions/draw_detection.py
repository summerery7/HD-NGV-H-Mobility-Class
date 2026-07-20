#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2


def draw_detection(img, x1, y1, x2, y2, uc, vc, label):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.circle(img, (int(uc), int(vc)), 4, (0, 0, 255), -1)

    y_text = int(y1) - 8 if y1 > 20 else int(y1) + 18
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (int(x1), y_text - th - 4),
                  (int(x1) + tw + 4, y_text + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (int(x1) + 2, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img
