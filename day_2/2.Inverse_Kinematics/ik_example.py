import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(ROOT, "8.xml")
XML_PATH = os.path.join(XML_DIR, "mg400.xml")
sys.path.insert(0, os.path.join(ROOT, "7.utils"))

from simple_fk import forward_kinematics
from simple_ik import inverse_kinematics
from video_logger import VideoLogger


p_input = np.array([0.152, 0.060, 0.230])


joint_names = ["MG400_Joint_1", "MG400_Joint_2", "MG400_Joint_3", "MG400_Joint_Gripper"]
passive_joint_names = ["MG400_Joint_6", "MG400_Joint_10", "MG400_Joint_6p",
                       "MG400_Joint_9", "MG400_Joint_5", "MG400_Joint_4"]


def load_model():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml_text = f.read()
    assets = {}
    for sub, prefix in (("meshes", "meshes/"), ("textures", "")):
        d = os.path.join(XML_DIR, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            with open(os.path.join(d, name), "rb") as f:
                assets[prefix + name] = f.read()
    return mujoco.MjModel.from_xml_string(xml_text, assets)


R2D = 180.0 / np.pi


def mj2dh_q(q):
    return np.array([-q[0], np.pi / 2 + q[2], np.pi + q[1], q[3]]) * R2D


def dh2mj_q(th_deg):
    return np.array([-th_deg[0], th_deg[2] - 180.0, th_deg[1] - 90.0]) / R2D


def world2dh_p(p, base):
    bx, bz = base
    return np.array([(bx - p[0]) * 1000.0, -p[1] * 1000.0, (p[2] - bz) * 1000.0])


def dh2world_p(p_mm, base):
    bx, bz = base
    return np.array([bx - p_mm[0] / 1000.0, -p_mm[1] / 1000.0, bz + p_mm[2] / 1000.0])


def calibrate_base(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p_w = data.site_xpos[model.site("gripper_magnet_site").id]
    p_dh, _ = forward_kinematics(mj2dh_q(np.zeros(4)))
    return (float(p_w[0] + p_dh[0] / 1000.0), float(p_w[2] - p_dh[2] / 1000.0))


def set_pose(model, data, qadr, p_qadr, p_dadr, q_active, iters=10):
    data.qpos[qadr] = q_active
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        ne = data.ne
        r = data.efc_pos[:ne].copy()
        if np.abs(r).max() < 1e-9:
            break
        J = data.efc_J[: ne * model.nv].reshape(ne, model.nv)[:, p_dadr]
        dq, *_ = np.linalg.lstsq(J, -r, rcond=None)
        data.qpos[p_qadr] += dq
    mujoco.mj_forward(model, data)


def move_to(model, data, viewer, qadr, p_qadr, p_dadr, q_goal, duration=3.0,
            video=None):
    q0 = data.qpos[qadr].copy()
    fps = 60
    n = int(duration * fps)
    for k in range(1, n + 1):
        tau = k / n
        s = 3.0 * tau**2 - 2.0 * tau**3
        set_pose(model, data, qadr, p_qadr, p_dadr, q0 + s * (q_goal - q0))
        viewer.sync()
        if video is not None:
            video.add(data)
        time.sleep(1.0 / fps)
    set_pose(model, data, qadr, p_qadr, p_dadr, q_goal)
    viewer.sync()


def main():
    model = load_model()
    data = mujoco.MjData(model)
    base = calibrate_base(model)

    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
    p_qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in passive_joint_names])
    p_dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in passive_joint_names])
    site = model.site("gripper_magnet_site").id


    p_dh = world2dh_p(p_input, base)
    th_dh = inverse_kinematics(p_dh)
    q_sol_rad = dh2mj_q(th_dh)
    q_sol_deg = np.rad2deg(q_sol_rad)


    p_check = dh2world_p(forward_kinematics(np.append(th_dh, 0.0))[0], base)
    reach_err = np.linalg.norm(p_check - p_input)

    p_home = dh2world_p(forward_kinematics(mj2dh_q(np.zeros(4)))[0], base)

    print("=" * 60)
    print(f"시작(홈) 자세   : 관절각 [0, 0, 0] deg → 말단 {np.round(p_home, 4)} m")
    print(f"목표 말단 위치  : {p_input} m")
    print(f"[IK 계산] 관절각: J1={q_sol_deg[0]:.2f}, J2={q_sol_deg[1]:.2f}, "
          f"J3={q_sol_deg[2]:.2f} deg")
    if reach_err > 1e-6:
        print(f"경고: 목표가 작업공간 밖입니다 — 가장 가까운 해로 이동 "
              f"(잔여 {reach_err * 1e3:.2f} mm)")
    print("=" * 60)
    print("뷰어에서 홈 자세 → 계산된 각도로 이동합니다...")


    q_goal = np.append(q_sol_rad, 0.0)
    video = VideoLogger(model, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "ik_example.mp4"), fps=60)
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        move_to(model, data, viewer, qadr, p_qadr, p_dadr, q_goal, video=video)
        for _ in range(60):
            video.add(data)
        video.close()


        p_meas = data.site_xpos[site].copy()
        print(f"[도달] 실측 말단 : {np.round(p_meas, 4)} m")
        print(f"[오차] |목표-실측|: {np.linalg.norm(p_input - p_meas) * 1e3:.2f} mm  (모델-실물 차이)")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
