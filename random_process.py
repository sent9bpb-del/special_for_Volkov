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
