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


def symbolic():
    import sympy as sp
    from simple_fk import symbolic as fk_symbolic
    pos, _ = fk_symbolic()
    t1, t2, t3p, t5 = sp.symbols('theta1 theta2 theta3p theta5', real=True)
    return sp.trigsimp(pos.jacobian([t1, t2, t3p, t5]))


if __name__ == "__main__":

    import sympy as sp
    Js = symbolic()
    print("=== [symbolic]  Jp = d p / d q  (q=theta1,theta2,theta3',theta5) ===")
    sp.pprint(Js, wrap_line=False)


    q = (0, 90, 150, 0)
    print("\n=== [numeric]  q(deg) =", q, "===")
    print("[JAC] Jp(3x4)[mm/rad] =\n", jacobian(q).round(3))
