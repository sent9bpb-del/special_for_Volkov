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


