# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vs_common import HandEyeVS, axis_angle, draw_overlay, Q_INIT, STANDOFF

LAMBDA = 1.5
MAX_STEPS = 5000
RENDER_EVERY = 10
TOL_T = 5e-4
TOL_R = np.radians(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="창 없이 실행")
    ap.add_argument("--video", default="", help="mp4 저장 경로 (예: pbvs.mp4)")
    args = ap.parse_args()

    sim = HandEyeVS()
    sim.reset(Q_INIT)

    R_des, p_des = sim.desired_cam_pose(STANDOFF)
    corners_w = sim.board_corners_world()
    des_px, _, _ = sim.project(corners_w, R_des, p_des)

    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 1.0 / (sim.dt * RENDER_EVERY), (sim.W * 2, sim.H))

    t_log, et_log, er_log = [], [], []
    step = 0
    for step in range(MAX_STEPS):
        R, p = sim.cam_pose()
        t_err = R.T @ (p_des - p)
        thu = axis_angle(R.T @ R_des)

        t_log.append(step * sim.dt)
        et_log.append(np.linalg.norm(t_err))
        er_log.append(np.linalg.norm(thu))

        if et_log[-1] < TOL_T and er_log[-1] < TOL_R:
            print(f"수렴: t={t_log[-1]:.2f}s, |t_err|={et_log[-1]*1000:.2f}mm, "
                  f"|R_err|={np.degrees(er_log[-1]):.3f}deg")
            break

        v_c = LAMBDA * t_err
        w_c = LAMBDA * thu
        qdot = sim.cam_twist_to_qdot(v_c, w_c)
        sim.step(qdot)

        if step % RENDER_EVERY == 0:
            cur_px, _, _ = sim.project(corners_w)
            ctr_px, _, _ = sim.project(sim.board_pose()[1][None, :])
            txt = f"PBVS t={t_log[-1]:5.2f}s  |t|={et_log[-1]*1000:6.1f}mm  |R|={np.degrees(er_log[-1]):5.1f}deg"
            eye = draw_overlay(sim.render("hand_eye"), cur_px, des_px, ctr_px[0], txt)
            over = cv2.cvtColor(sim.render("overview"), cv2.COLOR_RGB2BGR)
            frame = np.hstack([eye, over])
            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("PBVS  [left: hand-eye cam | right: overview]", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    else:
        print(f"최대 스텝 도달: |t_err|={et_log[-1]*1000:.2f}mm, "
              f"|R_err|={np.degrees(er_log[-1]):.3f}deg")

    if writer is not None:
        writer.release()


    cur_px, _, _ = sim.project(corners_w)
    ctr_px, _, _ = sim.project(sim.board_pose()[1][None, :])
    eye = draw_overlay(sim.render("hand_eye"), cur_px, des_px, ctr_px[0], "PBVS final")
    over = cv2.cvtColor(sim.render("overview"), cv2.COLOR_RGB2BGR)
    cv2.imwrite("pbvs_final.png", np.hstack([eye, over]))


    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(t_log, np.array(et_log) * 1000)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("|t_err| [mm]")
    ax[0].set_title("PBVS translation error"); ax[0].grid(True)
    ax[1].plot(t_log, np.degrees(er_log))
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("|theta*u| [deg]")
    ax[1].set_title("PBVS rotation error"); ax[1].grid(True)
    fig.tight_layout()
    fig.savefig("pbvs_error.png", dpi=120)
    print("저장: pbvs_final.png, pbvs_error.png")

    if not args.headless:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
