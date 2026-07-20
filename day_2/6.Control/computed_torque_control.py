import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(ROOT, "8.xml")
XML_PATH = os.path.join(XML_DIR, "mg400.xml")
sys.path.insert(0, os.path.join(ROOT, "7.utils"))


from simple_fk import forward_kinematics
from simple_ik import inverse_kinematics
from simple_jac import jacobian
import simple_id
from video_logger import VideoLogger


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


dt: float = 0.0005
dt_physics: float = 0.0001

joint_names = ["MG400_Joint_1", "MG400_Joint_2", "MG400_Joint_3", "MG400_Joint_Gripper"]
actuator_names = ["MG400_Joint_1", "MG400_Joint_2", "MG400_Joint_3", "M0G400_Joint_Gripper"]
joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Gripper"]


center = np.array([0.0, -0.25, 0.15, 0.0])
amplitude = np.array([0.5, 0.15, 0.15, 0.5])
freq = np.array([0.15, 0.15, 0.15, 0.10])
phase = np.array([0.0, 0.0, 0.0, 0.0])
run_duration = 10.0


def desired_trajectory(t: float):
    w = 2 * np.pi * freq
    arg = w * t + phase
    q_d = center + amplitude * np.sin(arg)
    v_d = amplitude * w * np.cos(arg)
    a_d = -amplitude * w**2 * np.sin(arg)
    return q_d, v_d, a_d


Kp = np.array([300.0, 3000.0, 5000.0, 50.0])
Kv = 2.0 * np.sqrt(Kp)

settle_time = 2.0


R2D = 180.0 / np.pi


def mj2dh_q(q):
    return np.array([-q[0], np.pi / 2 + q[2], np.pi + q[1], q[3]]) * R2D


def dh2mj_q(th_deg):
    return np.array([-th_deg[0], th_deg[2] - 180.0, th_deg[1] - 90.0]) / R2D


def swap_dh_mj(v3):
    return np.array([-v3[0], v3[2], v3[1]])


def world2dh_p(p, base):
    bx, bz = base
    return np.array([(bx - p[0]) * 1000.0, -p[1] * 1000.0, (p[2] - bz) * 1000.0])


def dh2world_p(p_mm, base):
    bx, bz = base
    return np.array([bx - p_mm[0] / 1000.0, -p_mm[1] / 1000.0, bz + p_mm[2] / 1000.0])


def setup_controller(model):

    groups = (["MG400_Link_2", "MG400_Link_3", "MG400_Link_4", "MG400_Link_5"],
              ["MG400_Link_6", "MG400_Link_6p", "MG400_Link_9"],
              ["MG400_Link_10", "MG400_Link_12", "MG400_Gripper"])
    masses = tuple(float(sum(model.body(n).mass for n in g)) for g in groups)
    simple_id.set_physical_params(masses, grav=-model.opt.gravity[2])
    print("XML 물성치 → DH 링크 질량 [kg]:", np.round(masses, 4))


    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    p_w = data.site_xpos[model.site("gripper_magnet_site").id]
    p_dh, _ = forward_kinematics(mj2dh_q(np.zeros(4)))
    base = (float(p_w[0] + p_dh[0] / 1000.0), float(p_w[2] - p_dh[2] / 1000.0))


    I1 = axis_inertia(model, "MG400_Link_1", "MG400_Joint_1")
    Ig = axis_inertia(model, "MG400_Gripper", "MG400_Joint_Gripper")
    I_extra = np.array([I1, 0.0, 0.0, Ig])
    return base, I_extra


def quat2mat_np(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_inertia(model, body_name, joint_name):
    b = model.body(body_name)
    R = quat2mat_np(model.body_iquat[b.id])
    I_body = R @ np.diag(model.body_inertia[b.id]) @ R.T
    a = model.jnt_axis[model.joint(joint_name).id]
    a = a / np.linalg.norm(a)
    return float(a @ I_body @ a)


def ctm_torque(q, v, u, I_extra):
    th = np.deg2rad(mj2dh_q(q)[:3])
    dth = swap_dh_mj(v[:3])
    ddth = swap_dh_mj(u[:3])

    Dm, Cv, Gv = simple_id.dynamics_terms(th, dth)

    tau_inertia = np.append(swap_dh_mj(Dm @ ddth), 0.0) + I_extra * u
    tau_coriolis = np.append(swap_dh_mj(Cv), 0.0)
    tau_gravity = np.append(swap_dh_mj(Gv), 0.0)
    tau = tau_inertia + tau_coriolis + tau_gravity
    return tau, (tau_inertia, tau_coriolis, tau_gravity)


def run_simulation(model, desired_fn, total_time, base, I_extra, cartesian=None, label=""):
    data = mujoco.MjData(model)
    model.opt.timestep = dt_physics
    n_sub = int(round(dt / dt_physics))

    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in joint_names])
    dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])
    force_lo, force_hi = model.actuator_forcerange[actuator_ids].T

    log = {k: [] for k in ("t", "q_d", "v_d", "a_d", "q", "v", "a",
                           "tau", "tau_inertia", "tau_coriolis", "tau_gravity",
                           "p_d", "p")}

    video_name = f"ctm_{label}.mp4" if label else "ctm_joint_sin.mp4"
    video = VideoLogger(model, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            video_name), fps=30)

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        q_d_settle = desired_fn(0.0)[0]

        while viewer.is_running():
            if total_time > 0 and data.time >= total_time + settle_time:
                break
            step_start = time.time()


            q = data.qpos[qadr].copy()
            v = data.qvel[dadr].copy()


            if data.time < settle_time:
                t_traj = 0.0
                q_d, v_d, a_d = q_d_settle, np.zeros(4), np.zeros(4)
            else:
                t_traj = data.time - settle_time
                q_d, v_d, a_d = desired_fn(t_traj)
            e = q_d - q

            #--TODO--#


            tau, (tau_inertia, tau_coriolis, tau_gravity) = ctm_torque(q, v, u, I_extra)


            tau = np.clip(tau, force_lo, force_hi)
            data.ctrl[actuator_ids] = tau
            for _ in range(n_sub):
                mujoco.mj_step(model, data)


            log["t"].append(data.time)
            log["q_d"].append(q_d.copy());  log["v_d"].append(v_d.copy());  log["a_d"].append(a_d.copy())
            log["q"].append(q.copy());      log["v"].append(v.copy());      log["a"].append(data.qacc[dadr].copy())
            log["tau"].append(tau.copy())
            log["tau_inertia"].append(tau_inertia.copy())
            log["tau_coriolis"].append(tau_coriolis.copy())
            log["tau_gravity"].append(tau_gravity.copy())
            if cartesian is not None:
                log["p_d"].append(np.asarray(cartesian(t_traj)))
                p_dh = forward_kinematics(mj2dh_q(q))[0]
                log["p"].append(dh2world_p(p_dh, base))

            viewer.sync()
            video.capture(data)
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    video.close()

    if log["t"]:
        plot_tracking(log)
        if cartesian is not None:
            plot_cartesian(log, label)
        plot_torque_decomposition(log)
        plt.show()


def run_cartesian(traj, label=""):
    model = load_model()
    base, I_extra = setup_controller(model)

    print(f"[{label}] 말단 경로를 관절 궤적으로 변환 중 (simple_ik / simple_jac)...")
    n = int(round(traj.total_time / dt)) + 1
    q_tab = np.zeros((n, 4))
    v_tab = np.zeros((n, 4))
    for k in range(n):
        p_w, v_w, _ = traj(k * dt)
        th = inverse_kinematics(world2dh_p(p_w, base))
        q_tab[k, :3] = dh2mj_q(th)

        J = jacobian(np.append(th, 0.0))[:, :3]
        v_dh = np.array([-v_w[0], -v_w[1], v_w[2]]) * 1000.0
        v_tab[k, :3] = swap_dh_mj(np.linalg.solve(J, v_dh))
    a_tab = np.gradient(v_tab, dt, axis=0)
    print(f"변환 완료: {n} 스텝 ({traj.total_time:.1f} s)")

    def desired(t):
        k = min(int(round(t / dt)), len(q_tab) - 1)
        return q_tab[k], v_tab[k], a_tab[k]

    run_simulation(model, desired, traj.total_time, base, I_extra,
                   cartesian=lambda t: traj(t)[0], label=label)


def main() -> None:
    model = load_model()
    base, I_extra = setup_controller(model)
    run_simulation(model, desired_trajectory, run_duration, base, I_extra)


def plot_tracking(log: dict) -> None:
    t = np.array(log["t"])
    n = len(joint_names)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 8), sharex=True, squeeze=False)
    rows = [
        ("Position [rad]", np.array(log["q_d"]), np.array(log["q"])),
        ("Velocity [rad/s]", np.array(log["v_d"]), np.array(log["v"])),
        ("Acceleration [rad/s^2]", np.array(log["a_d"]), np.array(log["a"])),
    ]
    for row, (ylabel, desired, actual) in enumerate(rows):
        for col in range(n):
            ax = axes[row][col]
            ax.plot(t, desired[:, col], "--", color="tab:red", label="desired")
            ax.plot(t, actual[:, col], "-", color="tab:blue", alpha=0.85, label="actual")
            if row == 0:
                ax.set_title(joint_labels[col])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == n - 1:
                ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("MG400 computed-torque tracking: desired (--) vs. actual")
    fig.tight_layout()


def plot_cartesian(log: dict, label: str = "") -> None:
    t = np.array(log["t"])
    Pd = np.array(log["p_d"])
    P = np.array(log["p"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
    for i, lb in enumerate(["x", "y", "z"]):
        ax = axes[0][i]
        ax.plot(t, Pd[:, i], "--", color="tab:red", label="desired")
        ax.plot(t, P[:, i], "-", color="tab:blue", alpha=0.85, label="actual")
        ax.set_title(lb)
        ax.set_xlabel("time [s]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("position [m]")
            ax.legend(fontsize=8, loc="upper right")
    ax = axes[0][3]
    ax.plot(Pd[:, 1], Pd[:, 2], "--", color="tab:red", label="desired")
    ax.plot(P[:, 1], P[:, 2], "-", color="tab:blue", alpha=0.85, label="actual")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("path (y-z plane)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"End-effector Cartesian tracking ({label})")
    fig.tight_layout()


def plot_torque_decomposition(log: dict) -> None:
    t = np.array(log["t"])
    tau = np.array(log["tau"])
    inertia = np.array(log["tau_inertia"])
    coriolis = np.array(log["tau_coriolis"])
    gravity = np.array(log["tau_gravity"])
    n = len(joint_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharex=True, squeeze=False)
    for col in range(n):
        ax = axes[0][col]
        ax.plot(t, inertia[:, col], color="tab:blue", label=r"inertia $D\,\ddot q$")
        ax.plot(t, coriolis[:, col], color="tab:green", label=r"Coriolis $C$")
        ax.plot(t, gravity[:, col], color="tab:orange", label=r"gravity $G$")
        ax.plot(t, tau[:, col], color="k", lw=1.6, label=r"total $\tau$ (applied)")
        ax.set_title(joint_labels[col])
        ax.set_xlabel("time [s]")
        if col == 0:
            ax.set_ylabel("torque [Nm]")
        ax.grid(True, alpha=0.3)
        if col == n - 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(r"Computed-torque decomposition per joint: $\tau = D\,u + C + G$")
    fig.tight_layout()


if __name__ == "__main__":
    main()
