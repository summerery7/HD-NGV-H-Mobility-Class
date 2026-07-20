import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import json
import argparse
import numpy as np
import mujoco
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot6dof.xml")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_output")
W, H = 1280, 720
PATTERN = (9, 6)
SQUARE = 0.008

N_TARGET = 18
MAX_TRIES = 80
SEED = 3

Q_NOMINAL = np.array([0.0, 0.4, 0.8, 0.0, 0.4, 0.0])
Q_SCALE = np.array([0.25, 0.18, 0.25, 1.0, 0.45, 1.5])


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).ravel()
    return T


def T_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return T_from(R.T, -R.T @ t)


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class Viz:

    def __init__(self, show, video_path, fps=30):
        self.show = show
        self.sw, self.sh = 640, 360
        self.bar = 46
        size = (self.sw * 2, self.sh + self.bar)
        self.writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if self.show:
            cv2.namedWindow("6DOF robot-world/hand-eye data collection", cv2.WINDOW_AUTOSIZE)

    def frame(self, scene_rgb, cam_bgr, status, hold=1):
        left = cv2.resize(cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR), (self.sw, self.sh))
        right = cv2.resize(cam_bgr, (self.sw, self.sh))
        cv2.putText(left, "overview", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(right, "extern_cam (eye-to-hand)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.zeros((self.sh + self.bar, self.sw * 2, 3), np.uint8)
        canvas[:self.sh] = np.hstack([left, right])
        for k, line in enumerate(status):
            cv2.putText(canvas, line, (10, self.sh + 18 + 20 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        for _ in range(hold):
            self.writer.write(canvas)
        if self.show:
            cv2.imshow("6DOF robot-world/hand-eye data collection", canvas)
            if cv2.waitKey(1 if hold == 1 else 400) & 0xFF == 27:
                self.show = False
                cv2.destroyAllWindows()

    def close(self):
        self.writer.release()
        if self.show:
            cv2.destroyAllWindows()


def load_model(xml_path):
    try:
        return mujoco.MjModel.from_xml_path(xml_path)
    except ValueError:


        base = os.path.dirname(xml_path)
        with open(xml_path, encoding="utf-8") as f:
            xml = f.read()
        assets = {}
        for root, _, files in os.walk(base):
            for fn in files:
                if fn.lower().endswith((".png", ".stl", ".obj", ".msh")):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, base).replace(os.sep, "/")
                    with open(full, "rb") as f:
                        assets[rel] = f.read()
        return mujoco.MjModel.from_xml_string(xml, assets)


def set_q(model, data, q):
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)


def move_to(model, data, q_to, renderer, viz, status, sec=0.8, fps=30):
    q_from = data.qpos[:6].copy()
    n = max(int(sec * fps), 1)
    for k in range(1, n + 1):
        a = 0.5 - 0.5 * np.cos(np.pi * k / n)
        set_q(model, data, (1 - a) * q_from + a * np.asarray(q_to))
        renderer.update_scene(data, camera="overview")
        scene = renderer.render()
        renderer.update_scene(data, camera="extern_cam")
        cam = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        q_txt = " ".join(f"{x:+.2f}" for x in data.qpos[:6])
        viz.frame(scene, cam, list(status) + [f"moving...  q = [{q_txt}]"])


def base_T_gripper(model, data):
    bid_base = model.body("base").id
    bid_grip = model.body("link6").id
    Rb = data.xmat[bid_base].reshape(3, 3)
    pb = data.xpos[bid_base]
    Rg = data.xmat[bid_grip].reshape(3, 3)
    pg = data.xpos[bid_grip]
    return T_from(Rb.T @ Rg, Rb.T @ (pg - pb))


def collect_data(model, data, renderer, viz):
    rng = np.random.default_rng(SEED)
    imgs, corners_all, T_bg_all = [], [], []
    for i in range(MAX_TRIES):
        q = Q_NOMINAL + rng.uniform(-1, 1, 6) * Q_SCALE
        status = [f"try {i+1}/{MAX_TRIES}  captured: {len(corners_all)}/{N_TARGET}"]
        move_to(model, data, q, renderer, viz, status)

        renderer.update_scene(data, camera="extern_cam")
        rgb = renderer.render()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        cam_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
            imgs.append(gray)
            corners_all.append(corners)
            T_bg_all.append(base_T_gripper(model, data))
            cv2.drawChessboardCorners(cam_bgr, PATTERN, corners, found)
            cap_txt = f"CAPTURE #{len(corners_all)}  corners {PATTERN[0]}x{PATTERN[1]} DETECTED"
        else:
            cap_txt = "DETECTION FAILED -> skip"
        cv2.putText(cam_bgr, cap_txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0) if found else (0, 0, 255), 2, cv2.LINE_AA)

        renderer.update_scene(data, camera="overview")
        viz.frame(renderer.render(), cam_bgr, status + [cap_txt], hold=12)

        if len(corners_all) >= N_TARGET:
            break

    print(f"수집 완료: {len(corners_all)} 캡처 / {i+1} 시도")
    return imgs, corners_all, T_bg_all


def ground_truth(model, data):
    mujoco.mj_forward(model, data)
    bid_base = model.body("base").id
    T_world_base = T_from(data.xmat[bid_base].reshape(3, 3), data.xpos[bid_base])

    cid = model.cam("extern_cam").id
    R_wc = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    Z_GT = T_inv(T_world_base) @ T_from(R_wc, data.cam_xpos[cid])

    bid_grip = model.body("link6").id
    T_world_grip = T_from(data.xmat[bid_grip].reshape(3, 3), data.xpos[bid_grip])
    bid_board = model.body("checkerboard").id
    T_world_board = T_from(data.xmat[bid_board].reshape(3, 3), data.xpos[bid_board])
    T_grip_board = T_inv(T_world_grip) @ T_world_board

    cx, cy = (PATTERN[0] - 1) / 2.0 * SQUARE, (PATTERN[1] - 1) / 2.0 * SQUARE
    cand = []
    for axis_angle, o in ((None, (-cx, -cy)),
                          ("z", (cx, cy)),
                          ("x", (-cx, cy)),
                          ("y", (cx, -cy))):
        R = np.eye(3) if axis_angle is None else Rot.from_euler(axis_angle, np.pi).as_matrix()
        cand.append(T_grip_board @ T_from(R, (o[0], o[1], 0.0015)))

    fovy = model.cam("extern_cam").fovy[0]
    f = H / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    return cand, Z_GT, K


def pick_X_GT(X_candidates, X_est):
    return min(X_candidates, key=lambda X: rot_err_deg(X[:3, :3], X_est[:3, :3]))


def report(tag, T_est, T_GT):
    dt = (T_est[:3, 3] - T_GT[:3, 3]) * 1000
    print(f"{tag:<26}rot err {rot_err_deg(T_est[:3,:3], T_GT[:3,:3]):8.4f} deg,"
          f"  t err {np.linalg.norm(dt):7.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true",
                    help="실시간 창 없이 실행 (mp4 기록은 그대로 남김)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)

    X_GT_cand, Z_GT, K_GT = ground_truth(model, data)


    print("=== [1] 데이터 수집 (보드는 말단 부착, 카메라는 외부 고정) ===")
    viz = Viz(show=not args.headless, video_path=f"{OUT_DIR}/collection.mp4")
    imgs, corners_all, T_bg_all = collect_data(model, data, renderer, viz)
    viz.close()
    print(f"수집 과정 영상 저장: {OUT_DIR}/collection.mp4")
    n = len(corners_all)

    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE


    print("\n=== [2] 카메라 내부 파라미터 (cv2.calibrateCamera) ===")
    K0 = cv2.initCameraMatrix2D([objp] * n, corners_all, (W, H))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp] * n, corners_all, (W, H), K0, None, flags=flags)
    print(f"재투영 RMS: {rms:.4f} px")
    print(f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}"
          f"   (GT: fx=fy={K_GT[0,0]:.2f}, cx={K_GT[0,2]:.1f}, cy={K_GT[1,2]:.1f})")
    print("dist:", np.round(dist.ravel(), 5), "(GT: 전부 0)")


    T_cb_all = [T_from(cv2.Rodrigues(rv)[0], tv) for rv, tv in zip(rvecs, tvecs)]


    print("\n=== [4] AX=ZB 폐형해 (cv2.calibrateRobotWorldHandEye, LI) ===")
    R_g2b = [T_inv(T)[:3, :3] for T in T_bg_all]
    t_g2b = [T_inv(T)[:3, 3] for T in T_bg_all]
    R_b2c = [T_inv(T)[:3, :3] for T in T_cb_all]
    t_b2c = [T_inv(T)[:3, 3] for T in T_cb_all]

    R_Z, t_Z, R_X, t_X = cv2.calibrateRobotWorldHandEye(
        R_g2b, t_g2b, R_b2c, t_b2c,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
    X0 = T_from(R_X, t_X)
    Z0 = T_from(R_Z, t_Z)
    report("[LI] X=gripper_T_board", X0, pick_X_GT(X_GT_cand, X0))
    report("[LI] Z=base_T_cam", Z0, Z_GT)


    print("\n=== [5] 비선형 정밀화 (재투영 오차 최소화, 내부 파라미터 동시 추정) ===")
    def pack(TX, TZ):
        return np.hstack([Rot.from_matrix(TX[:3, :3]).as_rotvec(), TX[:3, 3],
                          Rot.from_matrix(TZ[:3, :3]).as_rotvec(), TZ[:3, 3],
                          [K[0, 0], K[0, 2], K[1, 2], dist.ravel()[0], dist.ravel()[1]]])

    def unpack(p):
        TX = T_from(Rot.from_rotvec(p[0:3]).as_matrix(), p[3:6])
        TZ = T_from(Rot.from_rotvec(p[6:9]).as_matrix(), p[9:12])
        Kp = np.array([[p[12], 0, p[13]], [0, p[12], p[14]], [0, 0, 1.0]])
        distp = np.array([p[15], p[16], 0.0, 0.0, 0.0])
        return TX, TZ, Kp, distp

    def residuals(p):
        TX, TZ, Kp, distp = unpack(p)
        TZ_inv = T_inv(TZ)
        res = []
        for T_bg, corners in zip(T_bg_all, corners_all):
            T_cam_board = TZ_inv @ T_bg @ TX
            rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
            proj, _ = cv2.projectPoints(objp, rvec, T_cam_board[:3, 3], Kp, distp)
            res.append((proj - corners).ravel())
        return np.concatenate(res)

    p0 = pack(X0, Z0)
    sol = least_squares(residuals, p0, method="lm")
    X_ref, Z_ref, K_ref, dist_ref = unpack(sol.x)
    rms_before = np.sqrt(np.mean(residuals(p0) ** 2))
    rms_after = np.sqrt(np.mean(sol.fun ** 2))
    print(f"재투영 RMS: {rms_before:.3f} px → {rms_after:.3f} px")
    print(f"fx={K_ref[0,0]:.2f} cx={K_ref[0,2]:.2f} cy={K_ref[1,2]:.2f}"
          f"   (GT: fx={K_GT[0,0]:.2f}, cx={K_GT[0,2]:.1f}, cy={K_GT[1,2]:.1f})")


    print("\n=== [6] 최종 결과 vs ground truth ===")
    X_GT = pick_X_GT(X_GT_cand, X_ref)
    print("X = gripper_T_board (추정):\n", np.round(X_ref, 5))
    print("X = gripper_T_board (GT):\n", np.round(X_GT, 5))
    report("X refined", X_ref, X_GT)
    print("Z = base_T_cam (추정):\n", np.round(Z_ref, 5))
    print("Z = base_T_cam (GT):\n", np.round(Z_GT, 5))
    report("Z refined", Z_ref, Z_GT)

    img = cv2.cvtColor(imgs[0], cv2.COLOR_GRAY2BGR)
    T_cam_board = T_inv(Z_ref) @ T_bg_all[0] @ X_ref
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    proj, _ = cv2.projectPoints(objp, rvec, T_cam_board[:3, 3], K_ref, dist_ref)
    for c, p in zip(corners_all[0].reshape(-1, 2), proj.reshape(-1, 2)):
        cv2.circle(img, tuple(c.astype(int)), 6, (0, 255, 0), 1)
        cv2.drawMarker(img, tuple(p.astype(int)), (0, 0, 255),
                       cv2.MARKER_CROSS, 8, 1)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    with open(f"{OUT_DIR}/reprojection_check.png", "wb") as f:
        f.write(buf.tobytes())

    json.dump({
        "K": K.tolist(), "dist": dist.ravel().tolist(),
        "K_refined": K_ref.tolist(), "dist_refined": dist_ref.tolist(),
        "K_GT": K_GT.tolist(),
        "gripper_T_board_refined": X_ref.tolist(),
        "gripper_T_board_GT": X_GT.tolist(),
        "base_T_cam_refined": Z_ref.tolist(),
        "base_T_cam_GT": Z_GT.tolist(),
        "closed_form": {"LI": {"X": X0.tolist(), "Z": Z0.tolist()}},
        "reproj_rms_px": rms_after,
        "n_views": n,
    }, open(f"{OUT_DIR}/robot_eye_result.json", "w"), indent=2)
    print(f"\n저장: {OUT_DIR}/robot_eye_result.json, {OUT_DIR}/reprojection_check.png")


if __name__ == "__main__":
    main()
