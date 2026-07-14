# -*- coding: utf-8 -*-

import numpy as np
from simple_fk import forward_kinematics


def jacobian(q_deg, eps=1e-6):

    q = np.array(q_deg, float)
    J = np.zeros((3, 4))
    for i in range(4):
        d = np.zeros(4); d[i] = np.rad2deg(eps)
        J[:, i] = (forward_kinematics(q + d)[0] - forward_kinematics(q - d)[0]) / (2*eps)
    return J


if __name__ == "__main__":
    q = (0, 90, 150, 0)
    print("=== [numeric]  q(deg) =", q, "===")
    print("[JAC] Jp(3x4)[mm/rad] =\n", jacobian(q).round(3))
