# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vs_common import HandEyeVS, draw_overlay, Q_INIT, STANDOFF

LAMBDA = 1.5
MAX_STEPS = 5000
RENDER_EVERY = 10
TOL_PX = 0.3


def interaction_matrix(sn, Z):
    L = np.zeros((2 * len(Z), 6))
    for i, ((x, y), z) in enumerate(zip(sn, Z)):
        L[2 * i]     = [-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y]
        L[2 * i + 1] = [0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="창 없이 실행")
    ap.add_argument("--video", default="", help="mp4 저장 경로 (예: ibvs.mp4)")
    args = ap.parse_args()

    sim = HandEyeVS()
    sim.reset(Q_INIT)

    corners_w = sim.board_corners_world()


    R_des, p_des = sim.desired_cam_pose(STANDOFF)
    des_px, des_sn, _ = sim.project(corners_w, R_des, p_des)
    s_des = des_sn.ravel()

    writer = None
    if args.video:
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 1.0 / (sim.dt * RENDER_EVERY), (sim.W * 2, sim.H))

    t_log, epx_log = [], []
    for step in range(MAX_STEPS):
        cur_px, cur_sn, Z = sim.project(corners_w)
        e = cur_sn.ravel() - s_des

        t_log.append(step * sim.dt)
        epx_log.append(np.linalg.norm(cur_px - des_px, axis=1).mean())

        if epx_log[-1] < TOL_PX:
            print(f"수렴: t={t_log[-1]:.2f}s, 평균 픽셀 오차={epx_log[-1]:.3f}px")
            break

        L = interaction_matrix(cur_sn, Z)
        vc = -LAMBDA * np.linalg.pinv(L) @ e
        qdot = sim.cam_twist_to_qdot(vc[:3], vc[3:])
        sim.step(qdot)

        if step % RENDER_EVERY == 0:
            ctr_px, _, _ = sim.project(sim.board_pose()[1][None, :])
            txt = f"IBVS t={t_log[-1]:5.2f}s  mean err={epx_log[-1]:6.1f}px"
            eye = draw_overlay(sim.render("hand_eye"), cur_px, des_px, ctr_px[0], txt)
            over = cv2.cvtColor(sim.render("overview"), cv2.COLOR_RGB2BGR)
            frame = np.hstack([eye, over])
            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("IBVS  [left: hand-eye cam | right: overview]", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    else:
        print(f"최대 스텝 도달: 평균 픽셀 오차={epx_log[-1]:.3f}px")

    if writer is not None:
        writer.release()


    cur_px, _, _ = sim.project(corners_w)
    ctr_px, _, _ = sim.project(sim.board_pose()[1][None, :])
    eye = draw_overlay(sim.render("hand_eye"), cur_px, des_px, ctr_px[0], "IBVS final")
    over = cv2.cvtColor(sim.render("overview"), cv2.COLOR_RGB2BGR)
    cv2.imwrite("ibvs_final.png", np.hstack([eye, over]))


    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_log, epx_log)
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean feature error [px]")
    ax.set_title("IBVS feature error"); ax.grid(True)
    fig.tight_layout()
    fig.savefig("ibvs_error.png", dpi=120)
    print("저장: ibvs_final.png, ibvs_error.png")

    if not args.headless:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
