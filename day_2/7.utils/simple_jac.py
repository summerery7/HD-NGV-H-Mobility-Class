# -*- coding: utf-8 -*-

import numpy as np
from simple_fk import forward_kinematics


def jacobian(q_deg, eps=1e-6):

    #--TODO--#


if __name__ == "__main__":
    q = (0, 90, 150, 0)
    print("=== [numeric]  q(deg) =", q, "===")
    print("[JAC] Jp(3x4)[mm/rad] =\n", jacobian(q).round(3))
