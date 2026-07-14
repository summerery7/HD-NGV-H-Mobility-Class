
import numpy as np


class CubicTrajectory:
    def __init__(self, waypoints, segment_times):
        self.wp = np.atleast_2d(np.asarray(waypoints, dtype=float))
        self.Ts = np.asarray(segment_times, dtype=float)
        assert len(self.wp) == len(self.Ts) + 1, "웨이포인트는 구간 수 + 1개"
        self.t_knot = np.concatenate([[0.0], np.cumsum(self.Ts)])
        self.total_time = self.t_knot[-1]


        self.a0, self.a1, self.a2, self.a3 = [], [], [], []
        for k in range(len(self.Ts)):
            p0, pf, T = self.wp[k], self.wp[k + 1], self.Ts[k]
            delta = pf - p0
            self.a0.append(p0)
            self.a1.append(np.zeros_like(p0))
            self.a2.append(3.0 * delta / T**2)
            self.a3.append(-2.0 * delta / T**3)

    def __call__(self, t):
        t = np.clip(t, 0.0, self.total_time)
        k = min(np.searchsorted(self.t_knot, t, side="right") - 1, len(self.Ts) - 1)
        tau = t - self.t_knot[k]
        p = self.a0[k] + self.a1[k] * tau + self.a2[k] * tau**2 + self.a3[k] * tau**3
        v = self.a1[k] + 2.0 * self.a2[k] * tau + 3.0 * self.a3[k] * tau**2
        a = 2.0 * self.a2[k] + 6.0 * self.a3[k] * tau
        return p, v, a


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import computed_torque_control as ctc


    waypoints = np.array([
        [0.152,  0.00, 0.260],
        [0.152,  0.06, 0.260],
        [0.152,  0.06, 0.220],
        [0.152, -0.06, 0.220],
        [0.152, -0.06, 0.260],
        [0.152,  0.00, 0.260],
    ])
    segment_times = [2.0, 2.0, 2.0, 2.0, 2.0]

    traj = CubicTrajectory(waypoints, segment_times)


    t = np.arange(0.0, traj.total_time + 1e-9, 0.001)
    P, V, A = zip(*(traj(ti) for ti in t))
    P, V, A = np.array(P), np.array(V), np.array(A)

    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, data, ylabel in zip(axes, (P, V, A),
                                ("Position [m]", "Velocity [m/s]", "Accel [m/s$^2$]")):
        for i, lb in enumerate(labels):
            ax.plot(t, data[:, i], label=lb)
        for tk in traj.t_knot:
            ax.axvline(tk, color="gray", lw=0.5, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Cubic polynomial trajectory (per-axis $a_0+a_1\\tau+a_2\\tau^2+a_3\\tau^3$)")
    fig.tight_layout()


    ctc.run_cartesian(traj, label="cubic")
