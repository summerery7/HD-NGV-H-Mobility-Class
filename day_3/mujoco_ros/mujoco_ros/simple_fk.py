# -*- coding: utf-8 -*-

import numpy as np


L1, L2, L3, L4 = 43.0, 175.0, 175.0, 66.0
d1, d5 = 0.0, 53.0
PHI = np.pi


def _dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [0.0,   sa,     ca,   d  ],
                     [0.0,  0.0,    0.0,  1.0 ]])


def frames(q_deg):
    t1, t2, t3p, t5 = np.deg2rad(q_deg)
    th3 = t3p - PHI - t2
    th4 = PHI - t3p
    Ts = [_dh(t1,  d1, L1,  np.pi/2),
          _dh(t2,  0.0, L2,  0.0),
          _dh(th3, 0.0, L3,  0.0),
          _dh(th4, 0.0, L4, -np.pi/2),
          _dh(t5, -d5, 0.0,  0.0)]
    A = [np.eye(4)]
    for T in Ts:
        A.append(A[-1] @ T)
    return A


def forward_kinematics(q_deg):
    T = frames(q_deg)[-1]
    pos = T[:3, 3]
    R = T[:3, :3]
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return pos, np.rad2deg([roll, pitch, yaw])


if __name__ == "__main__":
    q = (0, 90, 150, 0)
    pos, euler = forward_kinematics(q)
    print("=== [numeric]  q(deg) =", q, "===")
    print("[FK] pos(mm)    =", pos.round(3))
    print("[FK] euler(deg) =", euler.round(3))
