import sympy as sp
import numpy as np

# === [1] 심볼릭 변수 ===
q = sp.Symbol('q')
Ixx, Iyy, Izz, Ixy, Ixz, Iyz = sp.symbols('Ixx Iyy Izz Ixy Ixz Iyz')
m, l, g = sp.symbols('m l g')

print("\n=== [Step 0] Variables ===")
print(f"Symbols: q, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m, l, g")

# === [2] T01 ===
c = sp.cos(q)
s = sp.sin(q)

T01 = sp.Matrix([
    [c, -s, 0, 0],
    [s,  c, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1]
])

# === [3] Q1 ===
Q1 = sp.Matrix([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 0, 0],
    [0,  0, 0, 0]
])

# === [4] U11 = Q1 * T01 ===
U11 = ???

print("\n=== [Step 1] U11 = ??? ===")
sp.pprint(U11)

# === [5] J ===
I11 = (-Ixx + Iyy + Izz)/2
I22 = ( Ixx - Iyy + Izz)/2
I33 = ( Ixx + Iyy - Izz)/2

Icm = sp.Matrix([
    [I11, Ixy, Ixz],
    [Ixy, I22, Iyz],
    [Ixz, Iyz, I33]
])

r_bar = sp.Matrix([l/2, 0, 0])

# Homogeneous inertia matrix
J = sp.Matrix([
    [Icm[0,0], Icm[0,1], Icm[0,2], m * r_bar[0]],
    [Icm[1,0], Icm[1,1], Icm[1,2], m * r_bar[1]],
    [Icm[2,0], Icm[2,1], Icm[2,2], m * r_bar[2]],
    [m * r_bar[0], m * r_bar[1], m * r_bar[2], m]
])

print("\n=== [Step 2] J (homogeneous inertia) ===")
sp.pprint(J)

# === [6] D11 = Tr(U11 J U11ᵀ) ===
D11 = sp.trace(???)

print("\n=== [Step 3] D11 = ??? ===")
sp.pprint(D11)

D11_expanded = sp.simplify(sp.expand(D11))
print("\n=== [Step 4] D11 expanded ===")
sp.pprint(D11_expanded)

# === h-term ===
U111 = sp.diff(U11, q)
h1 = sp.trace(???)
h1_s = sp.simplify(h1)

print("\n=== [h-term] ===")
sp.pprint(h1_s)

# === C-term ===
g_vec = sp.Matrix([0, -g, 0, 0])
r = sp.Matrix([l, 0, 0, 1])  

C1 = ???
C1_s = sp.simplify(C1[0])

print("\n=== [C-term] ===")
sp.pprint(C1_s)






# === [11] NumPy 값과 비교 ===
# NumPy 기준
m_val = 1.0
L_val = 1.0
lc_val = L_val / 2
I_val = m_val * L_val**2 / 12
g_val = 9.81
q_input_deg = 45
q_input_rad = np.deg2rad(q_input_deg)

D_py = I_val + m_val * L_val**2
h_py = 0.0
C_py = m_val * L_val * g_val * np.cos(q_input_rad)

# 심볼릭 대입
subs_vals = {
    Ixx: I_val,
    Iyy: I_val,
    Izz: I_val + m_val * L_val**2,
    Ixy: 0,
    Ixz: 0,
    Iyz: 0,
    m: m_val,
    l: L_val,
    g: g_val,
    q: q_input_rad
}

D11_sym = float(D11_expanded.subs(subs_vals))
h1_sym = float(h1_s.subs(subs_vals))
C1_sym = float(C1_s.subs(subs_vals))

print(f"\n=== [Numeric Check @ q={q_input_deg}deg] ===")
print(f"D NumPy = {D_py:.6f}")
print(f"h NumPy = {h_py:.6f}")
print(f"C NumPy = {C_py:.6f}")

print(f"\n=== [Numeric Check @ q={q_input_deg}deg] ===")
print(f"D symbolic = {D11_sym:.6f}")
print(f"h symbolic = {h1_sym:.6f}")
print(f"C symbolic = {C1_sym:.6f}")