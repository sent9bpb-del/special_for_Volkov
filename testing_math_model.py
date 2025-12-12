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