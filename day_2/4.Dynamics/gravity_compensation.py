import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(ROOT, "8.xml")


XML_PATH = os.path.join(XML_DIR, "mg400_dyn.xml")
sys.path.insert(0, os.path.join(ROOT, "7.utils"))

import dobot_id
from video_logger import VideoLogger


mode = "gravity_comp"
q_start_deg = np.array([0.0, -15.0, 10.0])
u_input = np.array([0.0, 0.0, 0.0])
run_duration = 3.0


demo_damping = 0.05

dt: float = 0.0005
dt_physics: float = 0.0001

joint_names = ["MG400_Joint_1", "MG400_Joint_2", "MG400_Joint_3", "MG400_Joint_Gripper"]
actuator_names = ["MG400_Joint_1", "MG400_Joint_2", "MG400_Joint_3", "M0G400_Joint_Gripper"]
joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Gripper"]
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


def swap_dh_mj(v3):
    return np.array([-v3[0], v3[2], v3[1]])


def setup_dynamics(model):
    from simple_fk import frames

    groups = (["MG400_Link_2", "MG400_Link_3", "MG400_Link_4", "MG400_Link_5"],
              ["MG400_Link_6", "MG400_Link_6p", "MG400_Link_9"],
              ["MG400_Link_10", "MG400_Link_12", "MG400_Gripper"])


    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)


    A_home = [a.copy() for a in frames(mj2dh_q(np.zeros(4)))]
    for a in A_home:
        a[:3, 3] /= 1000.0
    p_w = data.site_xpos[model.site("gripper_magnet_site").id]
    bx = p_w[0] + A_home[-1][0, 3]
    bz = p_w[2] - A_home[-1][2, 3]
    R_wd = np.diag([-1.0, -1.0, 1.0])
    t_wd = np.array([bx, 0.0, -bz])


    p_sh = R_wd @ data.xpos[model.body("MG400_Link_3").id] + t_wd
    dz = p_sh[2] - A_home[1][2, 3]
    for a in A_home[1:]:
        a[2, 3] += dz

    masses, Jbars, rbars = [], [], []
    for fnum, g in zip(dobot_id.FRAME, groups):
        Rj = A_home[fnum][:3, :3]
        pj = A_home[fnum][:3, 3]
        m_tot, mc, P = 0.0, np.zeros(3), np.zeros((3, 3))
        for name in g:
            i = model.body(name).id
            m = float(model.body_mass[i])
            c_j = Rj.T @ (R_wd @ data.xipos[i] + t_wd - pj)
            R_ij = Rj.T @ R_wd @ data.ximat[i].reshape(3, 3)
            I_com = R_ij @ np.diag(model.body_inertia[i]) @ R_ij.T
            P += 0.5 * np.trace(I_com) * np.eye(3) - I_com + m * np.outer(c_j, c_j)
            mc += m * c_j
            m_tot += m
        Jb = np.zeros((4, 4))
        Jb[:3, :3] = P
        Jb[:3, 3] = mc
        Jb[3, :3] = mc
        Jb[3, 3] = m_tot
        masses.append(m_tot)
        Jbars.append(Jb)
        rbars.append(np.append(mc / m_tot, 1.0))

    dobot_id.set_physical_params(masses, grav=-model.opt.gravity[2],
                                  Jbars=Jbars, rbars=rbars)
    print("XML 물성치 → DH 링크 질량 [kg]:", np.round(masses, 4))


def dynamics_torque(q, v, u):
    th = np.deg2rad(mj2dh_q(q)[:3])
    dth = swap_dh_mj(v[:3])

    Dm, Cv, Gv = dobot_id.dynamics_terms(th, dth)

    tau_D = np.append(swap_dh_mj(Dm @ swap_dh_mj(u)), 0.0)
    tau_C = np.append(swap_dh_mj(Cv), 0.0)
    tau_G = np.append(swap_dh_mj(Gv), 0.0)
    return tau_D + tau_C + tau_G, (tau_D, tau_C, tau_G)


def true_torque(model, data, data_aux, dadr, p_dadr, u):
    nv = model.nv
    data_aux.qpos[:] = data.qpos
    data_aux.qvel[:] = data.qvel
    mujoco.mj_forward(model, data_aux)


    ne = data_aux.ne
    J = data_aux.efc_J[: ne * nv].reshape(ne, nv)
    G_p, *_ = np.linalg.lstsq(J[:, p_dadr], -J[:, dadr[:3]], rcond=None)

    data_aux.qacc[:] = 0.0
    data_aux.qacc[dadr[:3]] = u
    data_aux.qacc[p_dadr] = G_p @ u
    mujoco.mj_inverse(model, data_aux)


    fi = data_aux.qfrc_inverse
    tau = fi[dadr].copy()
    tau[:3] += G_p.T @ fi[p_dadr]
    return tau


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


def main():
    model = load_model()
    data = mujoco.MjData(model)
    data_aux = mujoco.MjData(model)
    model.opt.timestep = dt_physics
    n_sub = int(round(dt / dt_physics))
    setup_dynamics(model)

    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
    p_qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in passive_joint_names])
    p_dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in passive_joint_names])
    dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joint_names])
    actuator_ids = np.array([model.actuator(n).id for n in actuator_names])
    force_lo, force_hi = model.actuator_forcerange[actuator_ids].T
    jnt_lo, jnt_hi = model.jnt_range[[model.joint(n).id for n in joint_names[:3]]].T


    model.dof_damping[np.concatenate([dadr, p_dadr])] = demo_damping


    q0 = np.append(np.deg2rad(q_start_deg), 0.0)
    set_pose(model, data, qadr, p_qadr, p_dadr, q0)
    data.qvel[:] = 0.0


    tau0 = np.clip(dynamics_torque(q0, np.zeros(4), np.zeros(3))[0], force_lo, force_hi)
    for _ in range(int(0.5 / dt_physics)):
        data.ctrl[actuator_ids] = tau0
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    data.time = 0.0

    print("=" * 60)
    print(f"모드          : {mode}")
    print(f"시작 관절각   : J1={q_start_deg[0]:g}, J2={q_start_deg[1]:g}, "
          f"J3={q_start_deg[2]:g} deg")
    print(f"인가 토크     : τ = D·u + C + G  (u = {u_input} rad/s², 피드백 없음)"
          if mode == "gravity_comp" else "인가 토크     : τ = 0  (자유 낙하)")
    print("=" * 60)

    log_t, log_tau_model, log_tau_true = [], [], []
    log_tau_D, log_tau_C, log_tau_G = [], [], []

    video = VideoLogger(model, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            f"gravity_compensation_{mode}.mp4"), fps=30)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running():
            if data.time >= run_duration:
                break
            step_start = time.time()


            q = data.qpos[qadr].copy()
            v = data.qvel[dadr].copy()


            if (np.abs(v[:3]).max() > 3.0
                    or np.any(q[:3] < jnt_lo + 0.02) or np.any(q[:3] > jnt_hi - 0.02)):
                print(f"t = {data.time:.2f} s: 드리프트로 관절한계/과속 도달 — 관찰 종료")
                break


            if mode == "gravity_comp":
                tau, (tau_D, tau_C, tau_G) = dynamics_torque(q, v, u_input)
            else:
                tau = tau_D = tau_C = tau_G = np.zeros(4)


            tau_true = true_torque(model, data, data_aux, dadr, p_dadr,
                                   u_input if mode == "gravity_comp" else np.zeros(3))


            tau = np.clip(tau, force_lo, force_hi)
            data.ctrl[actuator_ids] = tau
            for _ in range(n_sub):
                mujoco.mj_step(model, data)

            log_t.append(data.time)
            log_tau_model.append(tau.copy())
            log_tau_true.append(tau_true.copy())
            log_tau_D.append(tau_D.copy())
            log_tau_C.append(tau_C.copy())
            log_tau_G.append(tau_G.copy())

            viewer.sync()
            video.capture(data)
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    video.close()


    if log_t:
        t = np.array(log_t)
        tau_m = np.array(log_tau_model)
        tau_r = np.array(log_tau_true)
        tau_D = np.array(log_tau_D)
        tau_C = np.array(log_tau_C)
        tau_G = np.array(log_tau_G)
        diff = np.abs(tau_m[:, :3] - tau_r[:, :3])
        print("모델-실제 토크 차이 |Δτ| 평균 [Nm]:", np.round(diff.mean(axis=0), 4),
              " / 최대:", np.round(diff.max(axis=0), 4))


        rows = [
            (r"total $\tau$ [Nm]",   None,   None),
            (r"inertia $D\,u$ [Nm]", tau_D, "tab:blue"),
            (r"Coriolis $C$ [Nm]",   tau_C, "tab:green"),
            (r"gravity $G$ [Nm]",    tau_G, "tab:orange"),
        ]
        fig, axes = plt.subplots(4, 3, figsize=(13, 10), sharex=True)
        for r, (ylabel, series, color) in enumerate(rows):
            for i in range(3):
                ax = axes[r][i]
                if r == 0:
                    ax.plot(t, tau_m[:, i], color="k", lw=1.6,
                            label=r"model $\tau = D\,u + C + G$")
                    ax.plot(t, tau_r[:, i], "--", color="tab:red",
                            label="actual (MuJoCo)")
                    ax.set_title(joint_labels[i])
                    if i == 0:
                        ax.legend(fontsize=8, loc="best")
                else:
                    ax.plot(t, series[:, i], color=color, lw=1.4)
                if i == 0:
                    ax.set_ylabel(ylabel)
                if r == 3:
                    ax.set_xlabel("time [s]")
                ax.set_ylim(-0.5, 0.5)
                ax.grid(True, alpha=0.3)
        fig.suptitle("Actuator torque decomposition: $\\tau = D\\,u + C + G$ (model) vs actual")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
