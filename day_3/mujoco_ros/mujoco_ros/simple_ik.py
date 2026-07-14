# -*- coding: utf-8 -*-

import numpy as np


L1, L2, L3, L4 = 43.0, 175.0, 175.0, 66.0
d1, d5 = 0.0, 53.0
PHI = np.pi


def inverse_kinematics(p_mm):

    px, py, pz = p_mm
    th1 = np.arctan2(py, px)
    R = np.hypot(px, py) - L1 - L4
    Z = pz - d1 + d5
    cos_th3 = np.clip((R**2 + Z**2 - L2**2 - L3**2) / (2*L2*L3), -1.0, 1.0)
    th3 = -np.arccos(cos_th3)
    th2 = np.arctan2(Z, R) - np.arctan2(L3*np.sin(th3), L2 + L3*np.cos(th3))
    th3p = th2 + th3 + PHI
    return np.rad2deg([th1, th2, th3p])

if __name__ == "__main__":
    p = (284.0, 0.0, 122.0)
    q = inverse_kinematics(p)
    print("=== [numeric]  p(mm) =", p, "===")
    print("[IK] q(deg) =", q.round(3))
