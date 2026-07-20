import numpy as np


# === [1] 파라미터 정의 ===
m = 1.0               # 링크 질량 [kg]
L = 1.0               # 링크 길이 [m]
lc = L / 2            # 질량중심까지 거리 [m]
I = m * L**2 / 12     # 링크 중심 기준 관성 [kg m^2]
g = 9.81              # 중력 가속도 [m/s^2]
B = 0.5               # 마찰 계수

Kp = 20.0             # PD 제어 Kp
Kd = 5.0              # PD 제어 Kd

# === 입력 [degree] ===
q_input_deg = 60
q_input = np.deg2rad(q_input_deg)
dq_input = 0.0   # 초기 각속도
ddq_input = 0.0  # 초기 각가속도

print(f"=== [Robot Parameters] ===")
print(f"Mass (m)        : {m:.4f} kg")
print(f"Link Length (L) : {L:.4f} m")
print(f"CoM Distance    : {lc:.4f} m")
print(f"Inertia (I)     : {I:.4f} kg m^2")
print(f"Gravity (g)     : {g:.4f} m/s^2")
print(f"Damping (B)     : {B:.4f}")
print(f"PD gains        : Kp = {Kp:.2f}, Kd = {Kd:.2f}")
print(f"Input q         : {q_input_deg:.1f} deg ({q_input:.4f} rad)\n")

# === [2] Dynamics 함수 정의 ===
# D term : 링크의 관성 모멘트
def D(q):
    return I + m * L**2

# h term : 코리올리/원심력
def h(q, dq):
    dD_dq = 0.0  # 1자유도: 상수값
    return 0.5 * dD_dq

# C term : 중력
def C(q):
    return m * g * L * np.cos(q)

# === [3] 다이나믹스 텀 확인 ===
# q_input에 대해 D, h, C, τ 값을 계산하여 확인

D_val = D(q_input)
h_val = h(q_input, dq_input)
C_val = C(q_input)

# τ = D * ddq + h * dq^2 + C
tau_val = D_val * ddq_input + h_val * dq_input**2 + C_val

print(f"=== [Dynamic Terms @ q = {q_input_deg:.1f} deg] ===")
print(f"Inertia D(q)     : {D_val:.4f}")
print(f"Coriolis h(q,dq) : {h_val:.4f}")
print(f"Gravity C(q)     : {C_val:.4f}")
print(f"Torque τ         : {tau_val:.4f}\n")

# === [4] 목표 궤적 함수 ===
# 진동하는 sin 궤적
def qd(t):
    return ???

def dqd(t):
    return ???

def ddqd(t):
    return ???

# === [5] 운동방정식 (EoM) 정의 ===
def dynamics(t, y, mode):
    """모드별 토크 -> ddq 계산"""
    q, dq = y
    if mode == 'pendulum':
        tau = 0  # 외력 X
    elif mode == 'gravity_hold':
        tau = ???
    elif mode == 'trajectory':
        tau = ???
    else:
        raise ValueError("Unknown mode")
    ddq = ???
    return [dq, ddq]




import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# === [6] 시뮬레이션 설정 ===
t_span = (0, 10)
t_eval = np.linspace(*t_span, 500)

# 초기 상태 (입력값 사용)
y0_traj = [0.0, 0.0]

# ODE Solver
sol_traj = solve_ivp(dynamics, t_span, y0_traj, t_eval=t_eval, args=('trajectory',))

# === [7] 애니메이션 설정 ===
fig, ax = plt.subplots(figsize=(6, 5))
title = '3) Trajectory Tracking (PD Control with Trace)'

ax.set_xlim(-L-0.2, L+0.2)
ax.set_ylim(-L-0.2, L+0.2)
ax.set_aspect('equal')

ax.axhline(0, color='gray', lw=1)
ax.axvline(0, color='gray', lw=1)

line, = ax.plot([], [], 'o-', lw=4, color='blue')
desired_line, = ax.plot([], [], '--', lw=2, color='red')
annot = ax.text(-L, L+0.1, '', fontsize=10, va='top', ha='left')

trajectory_x = []
trajectory_y = []

# === [8] 애니메이션 초기화 ===
def init():
    line.set_data([], [])
    desired_line.set_data([], [])
    annot.set_text('')
    trajectory_x.clear()
    trajectory_y.clear()
    return line, desired_line, annot

# === [9] 프레임별 애니메이션 ===
def animate(i):
    t = t_eval[i]
    q, dq = sol_traj.y[:, i]
    D_ = D(q)
    h_ = h(q, dq)
    C_ = C(q)
    tau = D_ * ddqd(t) + Kp * (qd(t) - q) + Kd * (dqd(t) - dq) + C_
    ddq = (tau - B * dq - h_ - C_) / D_

    x, y = L * np.cos(q), L * np.sin(q)
    line.set_data([0, x], [0, y])

    trajectory_x.append(x)
    trajectory_y.append(y)
    desired_line.set_data(trajectory_x, trajectory_y)

    annot.set_text(
        f"q = {q:.3f} rad\n"
        f"dq = {dq:.3f} rad/s\n"
        f"ddq = {ddq:.3f} rad/s²\n"
        f"D = {D_:.3f}\n"
        f"h = {h_:.3f}\n"
        f"C = {C_:.3f}\n"
        f"tau = {tau:.3f}"
    )

    return line, desired_line, annot

# === [10] 애니메이션 실행 ===
ani = FuncAnimation(fig, animate, frames=len(t_eval),
                    init_func=init, interval=20, blit=True)

plt.show()
