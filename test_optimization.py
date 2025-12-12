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

def coordinate_descent(x1, x2, f0, bounds,
                       eps=1e-4, max_iter=50,
                       passes=4, steps=25):

    x1_min, x1_max = bounds[0]
    x2_min, x2_max = bounds[1]

    path_x = [x1]
    path_y = [x2]

    def search_coord(coord_idx, left, right, x1, x2):
        base_left, base_right = left, right
        best_coord = None
        best_val = None

        for _ in range(passes):
            grid = np.linspace(left, right, steps)

            for val in grid:
                if coord_idx == 0:
                    val_f = f0(val, x2)
                else:
                    val_f = f0(x1, val)

                if best_val is None or val_f < best_val:
                    best_val = val_f
                    best_coord = val

            delta = (right - left) / (steps - 1)
            left = max(base_left, best_coord - delta)
            right = min(base_right, best_coord + delta)

        if coord_idx == 0:
            x1_new, x2_new = best_coord, x2
        else:
            x1_new, x2_new = x1, best_coord

        return x1_new, x2_new, best_val

    iters = 0

    for _ in range(max_iter):
        x1_old, x2_old = x1, x2

        x1, x2, _ = search_coord(0, x1_min, x1_max, x1, x2)
        path_x.append(x1)
        path_y.append(x2)

        x1, x2, _ = search_coord(1, x2_min, x2_max, x1, x2)
        path_x.append(x1)
        path_y.append(x2)

        iters += 1

        if np.hypot(x1 - x1_old, x2 - x2_old) < eps:
            break

    return x1, x2, iters, path_x, path_y

bounds = ((-10, 10), (-10, 10))

print("Решение для тестовой функции f(x1, x2) = x1^2 + x2^2:")
x1, x2, iters, path_x, path_y = coordinate_descent(-1.3, -7, test_f, bounds)

print(
    f"X* = ({x1:.4f}, {x2:.4f}); "
    f"f(X*) = {test_f(x1, x2):.4f}, "
    f"число итераций = {iters}"
)

draw_contour_cd(path_x, path_y, test_f, bounds)

print("\nРешение для математической модели: ")
bounds = ((800, 950), (2, 5))
T0, V0 = 830.0, 2.5
x1, x2, iters, path_x, path_y = coordinate_descent(T0, V0, hcho_objective, bounds)

print(
    f"X* = ({x1:.4f}, {x2:.4f}); "
    f"f(X*) = {pozitive_hcho_objective(x1, x2):.4f}, "
    f"число итераций = {iters}"
)

draw_contour_cd(path_x, path_y, pozitive_hcho_objective, bounds)
