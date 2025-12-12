import matplotlib.pyplot as plt
import numpy as np

R = 8.31
A1 = 50
E1 = 64000
A2 = 10
E2 = 44000
A3 = 10**6
E3 = 108000
A4 = 25 * 10**5
E4 = 112000

m_c1_vh = 0.28
m_vozd_vh = 0.28
ro = 1.6
V = 3
T = 900
mu_ch3oh = 0.032
mu_o2 = 0.032
mu_hcho = 0.030
mu_h2 = 0.002
t0 = 0
d_t = 0.01
eps = 0.01
m = m_c1_vh + m_vozd_vh

def k(A, E, T):
    k = A * np.exp(-E/(R*T))
    return k

def per_C(C_m, mu):
    C = (C_m * 100 * mu) / ro
    return C

def t_s(V):
    v = m / ro
    t_s = V/v
    return t_s

def f_C1(C1_vh, C1, C3, C4):
    C1 = (1/t_s(V)) * (C1_vh - C1) - 2*k(A1, E1, T)*C1*C3 - k(A2, E2, T)*C1 - 2*k(A3, E3, T)*C1*C3 - k(A4, E4, T)*C1*C4
    return C1


def f_C2(C2_vh, C2, C1, C3):
    C2 = (1 / t_s(V)) * (C2_vh - C2) + 2 * k(A1, E1, T)*C1*C3 + k(A2, E2, T)*C1
    return C2


def f_C3(C3_vh, C3, C1):
    C3 = (1 / t_s(V)) * (C3_vh - C3) - k(A1, E1, T)*C1*C3 - 3 * k(A3, E3, T)*C1*C3
    return C3


def f_C4(C4_vh, C4, C1):
    C4 = (1 / t_s(V)) * (C4_vh - C4) - k(A4, E4, T)*C1*C4 + k(A2, E2, T)*C1
    return C4

C1_vh = (m_c1_vh * ro) / (m * mu_ch3oh)
C3_vh = (m_vozd_vh * 0.22 * ro) / (m * mu_o2)
C2_vh = 0
C4_vh = 0

C1 = C1_vh
C2 = C2_vh
C3 = C3_vh
C4 = C4_vh

C1_array = [C1]
C2_array = [C2]
C3_array = [C3]
C4_array = [C4]
C1_per_array = [per_C(C1, mu_ch3oh)]
C2_per_array = [per_C(C2, mu_hcho)]
C3_per_array = [per_C(C3, mu_o2)]
C4_per_array = [per_C(C4, mu_h2)]
t_array = [t0]

print(f"Концентрация метанола(CH3OH) на входе: {C1_array[0]} моль/м^3, {C1_per_array[0]} %")
print(f"Концентрация кислорода(O2) на входе: {C3_array[0]} моль/м^3, {C3_per_array[0]} %")

while True:
    C1 += f_C1(C1_vh, C1_array[-1], C3_array[-1], C4_array[-1]) * d_t
    C2 += f_C2(C2_vh, C2_array[-1], C1_array[-1], C3_array[-1]) * d_t
    C3 += f_C3(C3_vh, C3_array[-1], C1_array[-1]) * d_t
    C4 += f_C4(C4_vh, C4_array[-1], C1_array[-1]) * d_t
    t0 += d_t

    C1_array.append(C1)
    C2_array.append(C2)
    C3_array.append(C3)
    C4_array.append(C4)
    C1_per_array.append(per_C(C1, mu_ch3oh))
    C2_per_array.append(per_C(C2, mu_hcho))
    C3_per_array.append(per_C(C3, mu_o2))
    C4_per_array.append(per_C(C4, mu_h2))
    t_array.append(t0)

    if (abs(C1_per_array[-1] - C1_per_array[-2]) < eps) and \
            (abs(C2_per_array[-1] - C2_per_array[-2]) < eps) and \
            (abs(C3_per_array[-1] - C3_per_array[-2]) < eps) and \
            (abs(C4_per_array[-1] - C4_per_array[-2]) < eps):
        break

    print(f"Концентрация формальдегида(HCHO) на выходе: {C2_array[-1]} моль/м^3, {C2_per_array[-1]} %")
    print(f"Время: {t_array[-1]} c")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(t_array, C1_per_array)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C1 (CH3OH), %')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.subplot(1, 3, 2)
    plt.plot(t_array, C3_per_array)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C3 (O2), %')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.subplot(1, 3, 3)
    plt.plot(t_array, C2_per_array)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C2 (HCHO), %')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()
