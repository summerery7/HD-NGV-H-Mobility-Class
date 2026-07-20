# -*- coding: utf-8 -*-

import numpy as np
from simple_fk import frames, L2, L3, L4


GRAV     = 9.8
MASS     = (0.50, 0.40, 0.15)
LINK_LEN = (L2/1000, L3/1000, L4/1000)
FRAME    = (2, 3, 4)

_MASS = {j: m for j, m in zip(FRAME, MASS)}


def _T0(q):
    A = frames((np.degrees(q[0]), np.degrees(q[1]), np.degrees(q[2]), 0.0))
    out = []
    for a in A:
        b = a.copy(); b[:3, 3] /= 1000.0
        out.append(b)
    return out


def _pseudo_inertia(m, L):

    return np.array([[m*L**2/3, 0.0, 0.0, -m*L/2],
                     [0.0,      0.0, 0.0,  0.0],
                     [0.0,      0.0, 0.0,  0.0],
                     [-m*L/2,   0.0, 0.0,  m]])


_J    = {j: _pseudo_inertia(m, L) for j, m, L in zip(FRAME, MASS, LINK_LEN)}
_RBAR = {j: np.array([-L/2, 0.0, 0.0, 1.0]) for j, L in zip(FRAME, LINK_LEN)}


def _U(q, j, i, eps=1e-6):

    dq = np.zeros(3); dq[i] = eps
    return (_T0(q + dq)[j] - _T0(q - dq)[j]) / (2*eps)


def _Ud(q, j, k, i, eps=1e-5):
    dq = np.zeros(3); dq[i] = eps
    return (_U(q + dq, j, k) - _U(q - dq, j, k)) / (2*eps)


def D(q):
    Dm = np.zeros((3, 3))
    for j in FRAME:
        U = [_U(q, j, i) for i in range(3)]
        for i in range(3):
            for k in range(3):
                Dm[i, k] += np.trace(U[k] @ _J[j] @ U[i].T)
    return Dm


def C(q, dq):
    Cv = np.zeros(3)
    for j in FRAME:
        U = [_U(q, j, i) for i in range(3)]
        for i in range(3):
            for k in range(3):
                for m in range(3):
                    Cv[i] += np.trace(_Ud(q, j, k, m) @ _J[j] @ U[i].T) * dq[k] * dq[m]
    return Cv


def G(q):
    gvec = np.array([0.0, 0.0, -GRAV, 0.0])
    Gv = np.zeros(3)
    for j in FRAME:
        for i in range(3):
            Gv[i] += -_MASS[j] * (gvec @ _U(q, j, i) @ _RBAR[j])
    return Gv


def inverse_dynamics(q_deg, dq, ddq=(0.0, 0.0, 0.0), return_terms=False):
    q   = np.deg2rad(q_deg)
    dq  = np.asarray(dq, float)
    ddq = np.asarray(ddq, float)
    tau_D = D(q) @ ddq
    tau_C = C(q, dq)
    tau_G = G(q)
    tau   = tau_D + tau_C + tau_G
    if return_terms:
        return tau, tau_D, tau_C, tau_G
    return tau


def symbolic(simplify=True):

    import sympy as sp

    th  = sp.symbols('theta1 theta2 theta3p', real=True)
    dth = sp.symbols('dtheta1 dtheta2 dtheta3p', real=True)
    q   = list(th)
    L1s, L2s, L3s, L4s, d1s = sp.symbols('L1 L2 L3 L4 d1', positive=True)
    m1, m2, m3, g = sp.symbols('m1 m2 m3 g', positive=True)
    mass = {2: m1, 3: m2, 4: m3}; Ln = {2: L2s, 3: L3s, 4: L4s}
    PHI = sp.pi

    def dh(t, d, a, al):
        return sp.Matrix([[sp.cos(t), -sp.sin(t)*sp.cos(al), sp.sin(t)*sp.sin(al), a*sp.cos(t)],
                          [sp.sin(t),  sp.cos(t)*sp.cos(al), -sp.cos(t)*sp.sin(al), a*sp.sin(t)],
                          [0, sp.sin(al), sp.cos(al), d], [0, 0, 0, 1]])
    th3 = q[2] - PHI - q[1]; th4 = PHI - q[2]
    Ts = [dh(q[0], d1s, L1s, sp.pi/2), dh(q[1], 0, L2s, 0),
          dh(th3, 0, L3s, 0), dh(th4, 0, L4s, -sp.pi/2)]
    A = [sp.eye(4)]; cur = sp.eye(4)
    for T in Ts:
        cur = cur * T; A.append(cur)
    Js = {j: sp.Matrix([[mass[j]*Ln[j]**2/3, 0, 0, -mass[j]*Ln[j]/2], [0, 0, 0, 0],
                        [0, 0, 0, 0], [-mass[j]*Ln[j]/2, 0, 0, mass[j]]]) for j in FRAME}
    rb = {j: sp.Matrix([-Ln[j]/2, 0, 0, 1]) for j in FRAME}
    U  = {(j, i): A[j].diff(q[i]) for j in FRAME for i in range(3)}
    gv = sp.Matrix([[0, 0, -g, 0]])
    Dm, Cc, Gg = sp.zeros(3, 3), sp.zeros(3, 1), sp.zeros(3, 1)
    for j in FRAME:
        for i in range(3):
            Gg[i] += -mass[j] * (gv * U[(j, i)] * rb[j])[0]
            for k in range(3):
                Dm[i, k] += (U[(j, k)] * Js[j] * U[(j, i)].T).trace()
                for m in range(3):
                    Cc[i] += (U[(j, k)].diff(q[m]) * Js[j] * U[(j, i)].T).trace() * dth[k] * dth[m]
    if simplify:
        Dm, Cc, Gg = sp.trigsimp(Dm), sp.trigsimp(Cc), sp.trigsimp(Gg)
    return Dm, Cc, Gg


if __name__ == "__main__":
    import sympy as sp
    q   = (0, 90, 150)
    dq  = (1.0, 0.5, -0.2)
    ddq = (2.0, 1.0, 0.5)
    qr  = np.deg2rad(q)


    Ds, Cs, Gs = symbolic()
    print("=== [symbolic] ===")
    print("D(q) Inertia term:")
    print("  D[0,0] =", sp.simplify(Ds[0, 0]))
    print("  D[1,1] =", sp.simplify(Ds[1, 1]))
    print("  D[1,2] = D[2,1] =", sp.simplify(Ds[1, 2]))
    print("  D[2,2] =", sp.simplify(Ds[2, 2]))
    print("  (D[0,1]=D[0,2]=0)")
    print("C(q,dq) Coriolis term:")
    for i in range(3): print(f"  C[{i}] =", sp.simplify(Cs[i]))
    print("G(q) Gravity term:")
    for i in range(3): print(f"  G[{i}] =", sp.simplify(Gs[i]))


    ddth = sp.symbols('ddtheta1 ddtheta2 ddtheta3p', real=True)
    print("tau(q,dq,ddq) = D(q)*ddq + C(q,dq) + G(q):")
    for i in range(3):
        tau_D_i = sp.simplify(sum(Ds[i, k] * ddth[k] for k in range(3)))
        print(f"  tau_D[{i}] (Inertia, D*ddq) =", tau_D_i)
    for i in range(3):
        tau_i = sum(Ds[i, k] * ddth[k] for k in range(3)) + Cs[i] + Gs[i]
        print(f"  tau[{i}] =", tau_i)


    print("\n=== [numeric]  q(deg) =", q, "===")
    print("D(q) =\n", np.round(D(qr), 6))
    print("C(q,dq)      =", np.round(C(qr, dq), 6))
    print("G(q)         =", np.round(G(qr), 6))
    tau, tau_D, tau_C, tau_G = inverse_dynamics(q, dq, ddq, return_terms=True)
    print("\n--- torque terms (N·m) ---")
    print("D(q)·ddq (Inertia)  =", tau_D.round(4))
    print("C(q,dq)  (Coriolis) =", tau_C.round(4))
    print("G(q)     (Gravity)  =", tau_G.round(4))
    print("tau = D·ddq + C + G =", tau.round(4))
