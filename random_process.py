import numpy as np
import matplotlib.pyplot as plt
import math

lambda1 = math.pow(5, 9)
lambda2 = math.pow(3, 8)

M0 = 0.28
sigma0_2 = 26 * 10**(-4)
alpha0 = 0.09

N = 210
Kz_num = 20
Z0 = 1

A1 = 1
A2 = 1
Ns = 10

def generation_congurent_method(n, lam1, lam2, x0):
    x = [x0]
    for i in range(1, n):
        x.append((lam1 * x[i - 1]) % lam2)
    return [xi / lam2 - 0.5 for xi in x]

def checkmate_waiting(z):
    return sum(z) / len(z)

def dispersion(z, M):
    return sum((i - M) ** 2 for i in z) / len(z)

def correlation_function(z, M, Smax):
    N = len(z)
    K = []
    for S in range(Smax):
        ssum = 0
        for i in range(N - S):
            ssum += (z[i] - M) * (z[i + S] - M)
        K.append(ssum / (N - S))
    return K

def approximate_alpha(S, K, sig_2, max_iter=1000):
    alph = []
    for i in range(len(S)):
        alp = 1
        kt = sig_2 * math.exp(-alp * abs(S[i]))
        it = 0
        while kt < abs(K[i]) and it < max_iter:
            alp -= 0.1
            kt = sig_2 * math.exp(-alp * abs(S[i]))
            it += 1
        alph.append(max(alp, 0.001))

    alpha_final = sum(alph) / len(alph)
    k_appr = [sig_2 * math.exp(-alpha_final * abs(S[i])) for i in range(len(S))]
    return alpha_final, k_appr

def generate_process_z(x, A1, A2, sigma0_2, sigma_x0_2, alpha0, M0, Ns):
    z = []
    for k in range(len(x) - Ns):
        ssum = 0
        for i in range(k, k + Ns):
            ssum += (
                x[i]
                * math.sqrt(sigma0_2 / (sigma_x0_2 * alpha0 * A2))
                * A1
                * math.exp(-A2 * alpha0 * abs(i - k))
            )
        z.append(ssum / Ns + M0)
    return z

def simplex_search(x, sigma_x2, M0, sigma0_2, alpha0, Ns,
                   A1_init=1.0, A2_init=1.0,
                   tol=0.1, max_iter=200):

    def process_error(A1, A2):
        if A1 <= 0 or A2 <= 0:
            return 1e6
        z = generate_process_z(x, A1, A2, sigma0_2, sigma_x2, alpha0, M0, Ns)
        Mz = checkmate_waiting(z)
        sigma_z2 = dispersion(z, Mz)
        Kz = correlation_function(z, Mz, Kz_num)
        alpha_z, _ = approximate_alpha(range(Kz_num), Kz, sigma_z2)
        return (
            abs((Mz - M0) / M0)
            + abs((sigma_z2 - sigma0_2) / sigma0_2)
            + abs((alpha_z - alpha0) / alpha0)
        )

    simplex = [
        np.array([A1_init, A2_init]),
        np.array([A1_init + 0.2, A2_init]),
        np.array([A1_init, A2_init + 0.2])
    ]

    values = [process_error(p[0], p[1]) for p in simplex]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best, worst, second_worst = simplex[0], simplex[-1], simplex[-2]
        centroid = (best + second_worst) / 2

        reflected = centroid + (centroid - worst)
        fr = process_error(reflected[0], reflected[1])

        if fr < values[-1]:
            simplex[-1], values[-1] = reflected, fr

        if np.std(values) < tol:
            break

    return simplex[0][0], simplex[0][1]


