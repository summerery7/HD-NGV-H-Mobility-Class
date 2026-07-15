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
q_input_deg = input("Input q : ")
q_input_deg = float(q_input_deg)

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
