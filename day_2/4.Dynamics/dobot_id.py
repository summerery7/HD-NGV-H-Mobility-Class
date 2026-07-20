# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "7.utils"))
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


def set_physical_params(masses, grav=None, Jbars=None, rbars=None):
    global MASS, GRAV, _MASS, _J, _RBAR
    MASS = tuple(float(m) for m in masses)
    if grav is not None:
        GRAV = float(grav)
    _MASS = {j: m for j, m in zip(FRAME, MASS)}
    if Jbars is None:
        _J = {j: _pseudo_inertia(m, L) for j, m, L in zip(FRAME, MASS, LINK_LEN)}
        _RBAR = {j: np.array([-L/2, 0.0, 0.0, 1.0]) for j, L in zip(FRAME, LINK_LEN)}
    else:
        _J = {j: np.asarray(Jb, float) for j, Jb in zip(FRAME, Jbars)}
        _RBAR = {j: np.asarray(rb, float) for j, rb in zip(FRAME, rbars)}


def _U(q, j, i, eps=1e-6):
    dq = np.zeros(3); dq[i] = eps
    #--TODO--#


def _Ud(q, j, k, i, eps=1e-5):
    dq = np.zeros(3); dq[i] = eps
    #--TODO--#


def D(q):
    Dm = np.zeros((3, 3))
    for j in FRAME:
        U = [_U(q, j, i) for i in range(3)]
        for i in range(3):
            for k in range(3):
                #--TODO--#
    return Dm


def C(q, dq):
    Cv = np.zeros(3)
    for j in FRAME:
        U = [_U(q, j, i) for i in range(3)]
        for i in range(3):
            for k in range(3):
                for m in range(3):
                    #--TODO--#
    return Cv


def G(q):
    gvec = np.array([0.0, 0.0, -GRAV, 0.0])
    Gv = np.zeros(3)
    for j in FRAME:
        for i in range(3):
            Gv[i] += -_MASS[j] * (gvec @ _U(q, j, i) @ _RBAR[j])
    return Gv


def inverse_dynamics(q_deg, dq, ddq=(0.0, 0.0, 0.0)):
    q   = np.deg2rad(q_deg)
    dq  = np.asarray(dq, float)
    ddq = np.asarray(ddq, float)
    return D(q) @ ddq + C(q, dq) + G(q)


def dynamics_terms(q, dq):
    dq = np.asarray(dq, float)
    Dm = np.zeros((3, 3))
    Cv = np.zeros(3)
    Gv = np.zeros(3)
    gvec = np.array([0.0, 0.0, -GRAV, 0.0])
    for j in FRAME:
        U = [_U(q, j, i) for i in range(3)]
        for i in range(3):
            Gv[i] += -_MASS[j] * (gvec @ U[i] @ _RBAR[j])
            for k in range(3):
                Dm[i, k] += np.trace(U[k] @ _J[j] @ U[i].T)
        for k in range(3):
            for m in range(3):
                A = _Ud(q, j, k, m) @ _J[j]
                for i in range(3):
                    Cv[i] += np.sum(A * U[i]) * dq[k] * dq[m]
    return Dm, Cv, Gv


if __name__ == "__main__":
    q   = (0, 90, 150)
    dq  = (1.0, 0.5, -0.2)
    ddq = (2.0, 1.0, 0.5)
    qr  = np.deg2rad(q)

    print("=== [numeric]  q(deg) =", q, "===")
    print("D(q) =\n", np.round(D(qr), 6))
    print("C(q,dq)      =", np.round(C(qr, dq), 6))
    print("G(q)         =", np.round(G(qr), 6))
    print("tau(N·m)     =", inverse_dynamics(q, dq, ddq).round(4))
