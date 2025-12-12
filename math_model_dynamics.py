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

mu_ch3oh = 0.032
mu_o2 = 0.032
mu_hcho = 0.030
mu_h2 = 0.002

m = m_c1_vh + m_vozd_vh

def k(A, E, T):
    return A * np.exp(-E / (R * T))

def per_C(C_mol_per_m3, mu):
    return (C_mol_per_m3 * 100 * mu) / ro

def solve_model(V, T, d_t=0.01, eps=0.01, max_steps=200000):
    C1_vh = (m_c1_vh * ro) / (m * mu_ch3oh)
    C3_vh = (m_vozd_vh * 0.22 * ro) / (m * mu_o2)
    C2_vh = 0.0
    C4_vh = 0.0

    C1 = C1_vh
    C2 = C2_vh
    C3 = C3_vh
    C4 = C4_vh
    t = 0.0

    v = m / ro
    tau = V / v

    k1 = k(A1, E1, T)
    k2 = k(A2, E2, T)
    k3 = k(A3, E3, T)
    k4 = k(A4, E4, T)