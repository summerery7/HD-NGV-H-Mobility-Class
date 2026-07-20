#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading


class DetectorThread(threading.Thread):
    def __init__(self, model, imgsz=640):
        super().__init__(daemon=True)
        self.model = model
        self.imgsz = imgsz

        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending = None
        self._result = None
        self._running = True
        self.infer_ms = 0.0


    def submit(self, rgb):
        with self._lock:
            self._pending = rgb
        self._event.set()

    def get(self):
        with self._lock:
            return self._result

    def stop(self):
        self._running = False
        self._event.set()


    def run(self):
        import time
        import torch

        while self._running:
            self._event.wait()
            self._event.clear()
            if not self._running:
                break

            with self._lock:
                rgb = self._pending
                self._pending = None
            if rgb is None:
                continue

            t0 = time.time()
            with torch.inference_mode():
                results = self.model(rgb, size=self.imgsz)
            det = results.xyxy[0].cpu().numpy()
            dt = (time.time() - t0) * 1000.0

            with self._lock:
                self._result = det
                self.infer_ms = dt
