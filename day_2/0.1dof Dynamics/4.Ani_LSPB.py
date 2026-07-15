import numpy as np
import matplotlib.pyplot as plt

# 공통 시뮬레이션 설정
dt = 0.01
T = 5.0
steps = int(T / dt)

# 목표 각도
theta_desire = 1.0

# -----------------------------
# 1. Critical Damping Trajectory
# -----------------------------
theta_cd = 0.0
theta_dot_cd = 0.0
wn = 2.0
zeta_cd = 1.0 # critical damping

theta_cd_log = []
time_log = []

X1 = theta_cd - theta_desire
X2 = theta_dot_cd
for i in range(steps):
    t = i * dt
    X1_dot = X2
    X2_dot = -X1 * wn**2 - 2 * zeta_cd * wn *  X2
    theta_dot_cd += X2_dot * dt
    theta_cd += theta_dot_cd * dt
    X1 = theta_cd - theta_desire
    X2 = theta_dot_cd
    theta_cd_log.append(theta_cd)
    time_log.append(t)

# -----------------------------
# 2. LSPB Trajectory
# -----------------------------
q0 = 0.0
qf = theta_desire
tb = 0.5
v_max = (qf - q0) / (T - tb)
a = v_max / tb

q_lspb = []
for t in time_log:
    if t < tb:
        q = q0 + 0.5 * a * t**2
    elif t <= T - tb:
        q = q0 + 0.5 * a * tb**2 + v_max * (t - tb)
    else:
        q = qf - 0.5 * a * (T - t)**2
    q_lspb.append(q)
    # theta_cd +nd(q)


# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_log, theta_cd_log, label='Critical Damping', linewidth=2)
plt.plot(time_log, q_lspb, label='LSPB', linestyle='--', linewidth=2)
plt.axhline(theta_desire, color='gray', linestyle=':', label='Target θ')
plt.title('Trajectory Comparison: Critical Damping vs. LSPB')
plt.xlabel('Time [s]')
plt.ylabel('Position θ')
plt.grid(True)
plt.legend()
plt.show()