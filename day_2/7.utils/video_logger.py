# -*- coding: utf-8 -*-
import mujoco
import imageio


class VideoLogger:

    def __init__(self, model, path, fps=30, width=1280, height=720,
                 azimuth=135.0, elevation=-20.0, distance=0.85,
                 lookat=(0.26, 0.0, 0.20)):

        if model.vis.global_.offwidth < width:
            model.vis.global_.offwidth = width
        if model.vis.global_.offheight < height:
            model.vis.global_.offheight = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation
        if distance is not None:
            self.cam.distance = distance
        if lookat is not None:
            self.cam.lookat[:] = lookat

        self.path = path
        self.fps = fps
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        self.next_t = 0.0
        self.n_frames = 0

    def add(self, data):
        self.renderer.update_scene(data, camera=self.cam)
        self.writer.append_data(self.renderer.render())
        self.n_frames += 1

    def capture(self, data):
        if data.time + 1e-12 >= self.next_t:
            self.add(data)
            self.next_t += 1.0 / self.fps

    def close(self):
        self.writer.close()
        self.renderer.close()
        print(f"[video] {self.n_frames} 프레임 → {self.path}")
