import numpy as np


class LSPBViaTrajectory:
    def __init__(self, waypoints, segment_times, accel=0.5):
        self.wp = np.atleast_2d(np.asarray(waypoints, dtype=float))
        self.Ts = np.asarray(segment_times, dtype=float)
        assert len(self.wp) == len(self.Ts) + 1, "웨이포인트는 구간 수 + 1개"
        self.A = float(accel)
        self.t_knot = np.concatenate([[0.0], np.cumsum(self.Ts)])
        self.total_time = float(self.t_knot[-1])

        n_seg = len(self.Ts)
        dim = self.wp.shape[1]
        d = np.diff(self.wp, axis=0)


        t_start = self._rest_blend_time(d[0],       self.Ts[0])
        t_end   = self._rest_blend_time(d[-1],      self.Ts[-1])


        v = np.zeros((n_seg, dim))
        v[0]  = d[0]  / (self.Ts[0]  - 0.5 * t_start)
        v[-1] = d[-1] / (self.Ts[-1] - 0.5 * t_end)
        for s in range(1, n_seg - 1):
            v[s] = d[s] / self.Ts[s]


        t_via = np.zeros(n_seg + 1)
        t_via[0]      = t_start
        t_via[n_seg]  = t_end
        for s in range(1, n_seg):
            dv = v[s] - v[s - 1]
            t_via[s] = np.max(np.abs(dv)) / self.A if np.max(np.abs(dv)) > 0 else 0.0


        t_lin = np.zeros(n_seg)
        t_lin[0]  = self.Ts[0]  - t_start        - 0.5 * t_via[1]
        t_lin[-1] = self.Ts[-1] - 0.5 * t_via[n_seg - 1] - t_end
        for s in range(1, n_seg - 1):
            t_lin[s] = self.Ts[s] - 0.5 * t_via[s] - 0.5 * t_via[s + 1]
        if np.any(t_lin < -1e-9):
            bad = np.where(t_lin < 0)[0]
            raise ValueError(
                f"직선 구간 {bad} 의 시간이 음수 → 블렌드가 겹침. "
                f"accel 을 키우거나(현재 {self.A}) 해당 segment_times 를 늘리세요.")


        phases = []
        phases.append((t_start, v[0] / t_start if t_start > 0 else np.zeros(dim)))
        phases.append((t_lin[0], np.zeros(dim)))
        for s in range(1, n_seg):
            dv = v[s] - v[s - 1]
            a_blend = dv / t_via[s] if t_via[s] > 0 else np.zeros(dim)
            phases.append((t_via[s], a_blend))
            phases.append((t_lin[s], np.zeros(dim)))
        phases.append((t_end, -v[-1] / t_end if t_end > 0 else np.zeros(dim)))


        self.ph_t0, self.ph_p0, self.ph_v0, self.ph_a = [], [], [], []
        t0, p0, v0 = 0.0, self.wp[0].copy(), np.zeros(dim)
        for dur, a in phases:
            self.ph_t0.append(t0); self.ph_p0.append(p0.copy())
            self.ph_v0.append(v0.copy()); self.ph_a.append(a.copy())
            p0 = p0 + v0 * dur + 0.5 * a * dur ** 2
            v0 = v0 + a * dur
            t0 = t0 + dur
        self.ph_t0 = np.array(self.ph_t0)
        self._t_end_final = t0

    def _rest_blend_time(self, delta, td):
        best = 0.0
        for dq in np.abs(delta):
            disc = td * td - 2.0 * dq / self.A
            t = td - np.sqrt(max(disc, 0.0))
            best = max(best, t)
        return best

    def __call__(self, t):
        t = float(np.clip(t, 0.0, self._t_end_final))
        k = int(np.searchsorted(self.ph_t0, t, side="right") - 1)
        k = max(0, min(k, len(self.ph_t0) - 1))
        tau = t - self.ph_t0[k]
        a = self.ph_a[k]
        p = self.ph_p0[k] + self.ph_v0[k] * tau + 0.5 * a * tau ** 2
        v = self.ph_v0[k] + a * tau
        return p, v, a.copy()


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

    traj = LSPBViaTrajectory(waypoints, segment_times, accel=0.5)


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
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Multi-point LSPB (via-point overfly, Craig)")
    fig.tight_layout()

    ctc.run_cartesian(traj, label="lspb_via")
