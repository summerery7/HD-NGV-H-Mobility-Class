# -*- coding: utf-8 -*-

import numpy as np


L1, L2, L3, L4 = 43.0, 175.0, 175.0, 66.0
d1, d5 = 0.0, 53.0
PHI = np.pi


def inverse_kinematics(p_mm):




    #--TODO--#




if __name__ == "__main__":
    p = (284.0, 0.0, 122.0)
    q = inverse_kinematics(p)
    print("=== [numeric]  p(mm) =", p, "===")
    print("[IK] q(deg) =", q.round(3))
