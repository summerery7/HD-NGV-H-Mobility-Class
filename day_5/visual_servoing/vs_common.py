# -*- coding: utf-8 -*-
import os

import numpy as np
import mujoco

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "robot6dof_handeye.xml")


Q_INIT = np.array([-0.0217, -0.5845, 1.6816, 0.4163, 0.4201, -1.3675])


STANDOFF = 0.25


def axis_angle(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-10:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * w / (2.0 * np.sin(th))


class HandEyeVS:
    CAM = "hand_eye"
    EE_BODY = "link6"
    BOARD_BODY = "checkerboard"
    BOARD_SITE = "board_center"
    W, H = 640, 480

    def __init__(self, xml_path=XML_PATH):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.CAM)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.EE_BODY)
        self.board_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BOARD_BODY)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.BOARD_SITE)


        fovy = np.radians(self.model.cam_fovy[self.cam_id])
        self.f = 0.5 * self.H / np.tan(fovy / 2.0)
        self.K = np.array([[self.f, 0, self.W / 2.0],
                           [0, self.f, self.H / 2.0],
                           [0, 0, 1.0]])


        h = 0.048
        self.corners_local = np.array([[-h, -h, 0.0015],
                                       [ h, -h, 0.0015],
                                       [ h,  h, 0.0015],
                                       [-h,  h, 0.0015]])

        self.renderer = mujoco.Renderer(self.model, self.H, self.W)
        self.dt = self.model.opt.timestep
        self.jnt_lo = self.model.jnt_range[:, 0].copy()
        self.jnt_hi = self.model.jnt_range[:, 1].copy()


    def reset(self, q0=Q_INIT):
        self.data.qpos[:] = q0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def cam_pose(self):
        R_mj = self.data.cam_xmat[self.cam_id].reshape(3, 3)

        R = R_mj @ np.diag([1.0, -1.0, -1.0])
        p = self.data.cam_xpos[self.cam_id].copy()
        return R, p

    def board_pose(self):
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        p = self.data.site_xpos[self.site_id].copy()
        return R, p

    def board_corners_world(self):
        Rb = self.data.xmat[self.board_id].reshape(3, 3)
        pb = self.data.xpos[self.board_id]
        return (Rb @ self.corners_local.T).T + pb


    def project(self, pts_w, R=None, p=None):
        if R is None:
            R, p = self.cam_pose()
        pc = (np.asarray(pts_w) - p) @ R
        Z = pc[:, 2]
        xn = pc[:, 0] / Z
        yn = pc[:, 1] / Z
        px = np.column_stack([self.f * xn + self.W / 2.0,
                              self.f * yn + self.H / 2.0])
        return px, np.column_stack([xn, yn]), Z

    def desired_cam_pose(self, dist=STANDOFF):
        Rb, pb = self.board_pose()
        n = Rb[:, 2]
        p = pb + dist * n
        z = -n
        up = np.array([0.0, 0.0, 1.0])
        y = -(up - (up @ z) * z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.column_stack([x, y, z]), p


    def cam_twist_to_qdot(self, v_c, w_c, damp=0.01, qdot_max=1.5):
        R, p = self.cam_pose()
        v_w = R @ v_c
        w_w = R @ w_c
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, p, self.ee_id)
        J = np.vstack([jacp, jacr])
        V = np.concatenate([v_w, w_w])
        qdot = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(6), V)
        m = np.abs(qdot).max()
        if m > qdot_max:
            qdot *= qdot_max / m
        return qdot

    def step(self, qdot):
        q = self.data.qpos + qdot * self.dt
        self.data.qpos[:] = np.clip(q, self.jnt_lo, self.jnt_hi)
        mujoco.mj_forward(self.model, self.data)


    def render(self, cam="hand_eye"):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()


def draw_overlay(img_rgb, cur_px=None, des_px=None, center_px=None, text=""):
    import cv2
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hw, hh = img.shape[1] // 2, img.shape[0] // 2
    cv2.drawMarker(img, (hw, hh), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    if des_px is not None:
        pts = des_px.astype(int)
        for i, (u, v) in enumerate(pts):
            cv2.circle(img, (u, v), 6, (0, 0, 255), 1)
            cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 0, 255), 1)
    if cur_px is not None:
        pts = cur_px.astype(int)
        for i, (u, v) in enumerate(pts):
            cv2.circle(img, (u, v), 4, (0, 255, 0), -1)
            cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 1)
    if center_px is not None:
        u, v = center_px.astype(int)
        cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    if text:
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
