# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

GRAV = 9.806


m1, m2 = 1.0, 0.8
l1     = 0.30
r1, r2 = 0.15, 0.15
I1zz, I2zz = 0.020, 0.015
Im1, Im2   = 0.010, 0.010
Fs1, Fs2   = 0.10, 0.08
Fv1, Fv2   = 0.15, 0.12

theta_true = np.array([
    I1zz + m2 * l1**2 + Im1,
    I2zz,
    m2 * r2 * l1,
    m1 * r1 + m2 * l1,
    m2 * r2,
    Im2,
    Fs1, Fs2,
    Fv1, Fv2,
])
PARAM_NAMES = ["I1zz+m2*l1^2+Im1", "I2zz", "m2*r2*l1", "m1*r1+m2*l1",
               "m2*r2", "Im2", "Fs1", "Fs2", "Fv1", "Fv2"]


def M_mat(q, th):
    c2 = np.cos(q[1])
    M11 = th[0] + th[1] + 2 * th[2] * c2
    M12 = th[1] + th[2] * c2
    M22 = th[1] + th[5]
    return np.array([[M11, M12],
                     [M12, M22]])


def C_times_qd(q, qd, th):
    s2 = np.sin(q[1])
    a = th[2] * s2
    return np.array([-a * qd[1] * (2 * qd[0] + qd[1]),
                      a * qd[0] ** 2])


def g_vec(q, th):
    c1  = np.cos(q[0])
    c12 = np.cos(q[0] + q[1])
    return np.array([-th[3] * GRAV * c1 - th[4] * GRAV * c12,
                     -th[4] * GRAV * c12])


def d_vec(qd, th):
    return np.array([th[6] * np.sign(qd[0]) + th[8] * qd[0],
                     th[7] * np.sign(qd[1]) + th[9] * qd[1]])


def forward_accel(q, qd, tau, th):
    rhs = tau - C_times_qd(q, qd, th) - g_vec(q, th) - d_vec(qd, th)
    return np.linalg.solve(M_mat(q, th), rhs)


def W2_momentum(q, qd):
    c2 = np.cos(q[1])
    dq1, dq2 = qd
    W = np.zeros((2, 10))
    W[0, 0] = dq1
    W[0, 1] = dq1 + dq2
    W[0, 2] = 2 * c2 * dq1 + c2 * dq2
    W[1, 1] = dq1 + dq2
    W[1, 2] = c2 * dq1
    W[1, 5] = dq2
    return W


def W1_momentum(q, qd):
    c1, c12 = np.cos(q[0]), np.cos(q[0] + q[1])
    s2 = np.sin(q[1])
    dq1, dq2 = qd
    W = np.zeros((2, 10))
    W[1, 2] = -s2 * dq1 * (dq1 + dq2)
    W[0, 3] = GRAV * c1
    W[0, 4] = GRAV * c12
    W[1, 4] = GRAV * c12
    W[0, 6] = -np.sign(dq1)
    W[1, 7] = -np.sign(dq2)
    W[0, 8] = -dq1
    W[1, 9] = -dq2
    return W


def tau_excite(t):
    return np.array([
        1.2 * np.sin(1.0 * t) + 0.6 * np.cos(3.7 * t) + 0.3 * np.sin(9.0 * t),
        0.8 * np.sin(1.6 * t) + 0.5 * np.cos(5.1 * t) + 0.25 * np.sin(11.0 * t),
    ])


def simulate(T=12.0, dt=5e-4):
    n = int(round(T / dt))
    q  = np.zeros(2)
    qd = np.zeros(2)
    log = {k: np.zeros((n, 2)) for k in ("q", "qd", "tau")}
    log["t"] = np.zeros(n)

    def deriv(state, tau):
        qq, vv = state[:2], state[2:]
        return np.concatenate([vv, forward_accel(qq, vv, tau, theta_true)])

    for k in range(n):
        t = k * dt
        tau = tau_excite(t)
        s = np.concatenate([q, qd])
        k1 = deriv(s, tau)
        k2 = deriv(s + 0.5 * dt * k1, tau)
        k3 = deriv(s + 0.5 * dt * k2, tau)
        k4 = deriv(s + dt * k3, tau)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        log["t"][k]   = t
        log["q"][k]   = q
        log["qd"][k]  = qd
        log["tau"][k] = tau
        q, qd = s[:2], s[2:]
    return log


def estimate_momentum_batch(log, dt):
    W1_int = np.zeros((2, 10))
    u_int  = np.zeros(2)
    rows, rhs = [], []
    for k in range(len(log["t"])):
        q, qd, tau = log["q"][k], log["qd"][k], log["tau"][k]
        W1_int = W1_int + W1_momentum(q, qd) * dt
        u_int  = u_int + tau * dt
        rows.append(W2_momentum(q, qd) - W1_int)
        rhs.append(u_int.copy())
    Y = np.vstack(rows)
    u = np.concatenate(rhs)
    theta, *_ = np.linalg.lstsq(Y, u, rcond=None)
    return theta


def estimate_momentum_rls(log, dt, P0=100.0):
    W1_int = np.zeros((2, 10))
    u_int  = np.zeros(2)
    theta  = np.zeros(10)
    P      = P0 * np.eye(10)
    hist   = np.zeros((len(log["t"]), 10))
    I2 = np.eye(2)
    for k in range(len(log["t"])):
        q, qd, tau = log["q"][k], log["qd"][k], log["tau"][k]
        W1_int = W1_int + W1_momentum(q, qd) * dt
        u_int  = u_int + tau * dt
        Y = W2_momentum(q, qd) - W1_int
        S = I2 + Y @ P @ Y.T
        K = P @ Y.T @ np.linalg.inv(S)
        theta = theta + K @ (u_int - Y @ theta)
        P = P - K @ Y @ P
        hist[k] = theta
    return theta, hist


def main():
    dt = 5e-4
    log = simulate(T=12.0, dt=dt)


    th_mom = estimate_momentum_batch(log, dt)
    th_rls, hist = estimate_momentum_rls(log, dt)

    def err(th):
        return 100.0 * np.abs(th - theta_true) / np.abs(theta_true)

    print("=" * 62)
    print(f"{'param':<18}{'true':>10}{'momentum':>12}{'RLS':>11}")
    print("-" * 62)
    for i, nm in enumerate(PARAM_NAMES):
        print(f"{nm:<18}{theta_true[i]:>10.4f}{th_mom[i]:>12.4f}"
              f"{th_rls[i]:>11.4f}")
    print("-" * 62)
    print(f"{'mean |err| %':<18}{'':>10}{np.mean(err(th_mom)):>12.3f}"
          f"{np.mean(err(th_rls)):>11.3f}")
    print("=" * 62)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
    })


    fig1, ax = plt.subplots(figsize=(12, 6.8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        ax.plot(log["t"], hist[:, i], color=colors[i], lw=3.0,
                label=PARAM_NAMES[i])
        ax.axhline(theta_true[i], color=colors[i], ls="--", lw=1.8, alpha=0.6)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$\hat\theta$")
    ax.set_title("Momentum-based RLS: online parameter convergence "
                 "(dashed = true)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              framealpha=0.9, fontsize=12.5)
    ax.grid(True, alpha=0.3, linewidth=1.2)
    fig1.tight_layout()
    fig1.savefig(os.path.join(SAVE_DIR, "rls_convergence.png"), dpi=200,
                 bbox_inches="tight")


    fig2, ax = plt.subplots(figsize=(12, 6.2))
    x = np.arange(10); w = 0.35
    ax.bar(x - 0.5 * w, theta_true, w, label="true", color="k")
    ax.bar(x + 0.5 * w, th_mom,     w, label="momentum regressor")
    ax.set_xticks(x); ax.set_xticklabels(PARAM_NAMES, rotation=35, ha="right",
                                         fontsize=14)
    ax.set_ylabel(r"$\theta$"); ax.set_title("Estimated vs. true parameters")
    ax.legend(fontsize=15); ax.grid(True, alpha=0.3, axis="y", linewidth=1.2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, "param_bars.png"), dpi=200)

    print("plots saved: rls_convergence.png, param_bars.png")
    plt.show()


if __name__ == "__main__":
    main()
