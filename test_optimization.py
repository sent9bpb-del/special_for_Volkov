import numpy as np
import matplotlib.pyplot as plt
from math_model_dynamics import hcho_objective, pozitive_hcho_objective

def test_f(x1, x2):
    return x1**2 + x2**2

def draw_contour_cd(path_x, path_y, f0, bounds, grid_n=30):
    x1_min, x1_max = bounds[0]
    x2_min, x2_max = bounds[1]

    x1 = np.linspace(x1_min - 10, x1_max + 10, grid_n)
    x2 = np.linspace(x2_min - 1, x2_max + 1, grid_n)
    X1, X2 = np.meshgrid(x1, x2)

    Z = np.zeros_like(X1, dtype=float)
    for i in range(grid_n):
        for j in range(grid_n):
            T_val = X1[j, i]
            V_val = X2[j, i]
            Z[j, i] = f0(T_val, V_val)

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X1, X2, Z, levels=20)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.plot(path_x, path_y, 'o-', linewidth=1.5)

    plt.scatter(path_x[0], path_y[0],
                color='lime', s=80, edgecolor='black',
                zorder=3, label='start')
    plt.scatter(path_x[-1], path_y[-1],
                color='red', s=80, edgecolor='black',
                zorder=3, label='finish')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Покоординатный спуск')
    plt.grid(True)
    plt.legend()
    plt.show()