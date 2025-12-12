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