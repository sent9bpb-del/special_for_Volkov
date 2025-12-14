import matplotlib.pyplot as plt
import numpy as np
from random_process import z

# --- Константы и параметры модели ---

R = 8.31        # универсальная газовая постоянная, Дж/(моль·К)
A1 = 50
E1 = 64000      # Дж/моль
A2 = 10
E2 = 44000
A3 = 10**6
E3 = 108000
A4 = 25 * 10**5
E4 = 112000

m_c1_vh = z     # (кг/с) случайный процесс
m_vozd_vh = 0.28    # кг/с

ro = 1.6            # кг/м^3

mu_ch3oh = 0.032    # кг/моль
mu_o2 = 0.032    # кг/моль
mu_hcho = 0.030    # кг/моль
mu_h2 = 0.002    # кг/моль




def k(A, E, T):
    """Константа скорости по Аррениусу."""
    return A * np.exp(-E / (R * T))


def per_C(C_mol_per_m3, mu):
    """Перевод концентрации из моль/м^3 в % по массе."""
    return (C_mol_per_m3 * 100 * mu) / ro


def solve_model(V, T, m_c1_vh, d_t=0.01, eps=0.01, max_steps=200000):
    """
    Решение динамической модели для заданных V (м^3) и T (K).

    Возвращает словарь с массивами концентраций, времени и
    значениями на выходе.
    """
    m = m_c1_vh + m_vozd_vh  # суммарный массовый расход, кг/с

    # Входные концентрации (моль/м^3)
    C1_vh = (m_c1_vh * ro) / (m * mu_ch3oh)          # CH3OH
    C3_vh = (m_vozd_vh * 0.22 * ro) / (m * mu_o2)    # O2 (22% в воздухе)
    C2_vh = 0.0                                      # HCHO
    C4_vh = 0.0                                      # H2

    # Начальные значения (на входе в аппарат)
    C1 = C1_vh
    C2 = C2_vh
    C3 = C3_vh
    C4 = C4_vh
    t = 0.0

    # Гидродинамика
    v = m / ro          # объёмный расход, м^3/с
    tau = V / v         # время пребывания, с

    # Константы скоростей реакций при данной T
    k1 = k(A1, E1, T)
    k2 = k(A2, E2, T)
    k3 = k(A3, E3, T)
    k4 = k(A4, E4, T)

    # Массивы для хранения решения
    C1_array = [C1]
    C2_array = [C2]
    C3_array = [C3]
    C4_array = [C4]

    C1_per_array = [per_C(C1, mu_ch3oh)]
    C2_per_array = [per_C(C2, mu_hcho)]
    C3_per_array = [per_C(C3, mu_o2)]
    C4_per_array = [per_C(C4, mu_h2)]
    t_array = [t]

    # Интегрирование до стационарного режима
    for step in range(max_steps):

        # правая часть ДУ (производные dC/dt)
        dC1dt = (1 / tau) * (C1_vh - C1) - 2 * k1 * C1 * C3 - k2 * C1 - 2 * k3 * C1 * C3 - k4 * C1 * C4
        dC2dt = (1 / tau) * (C2_vh - C2) + 2 * k1 * C1 * C3 + k2 * C1
        dC3dt = (1 / tau) * (C3_vh - C3) - k1 * C1 * C3 - 3 * k3 * C1 * C3
        dC4dt = (1 / tau) * (C4_vh - C4) - k4 * C1 * C4 + k2 * C1

        # шаг Эйлера
        C1 += dC1dt * d_t
        C2 += dC2dt * d_t
        C3 += dC3dt * d_t
        C4 += dC4dt * d_t
        t += d_t

        # записываем
        C1_array.append(C1)
        C2_array.append(C2)
        C3_array.append(C3)
        C4_array.append(C4)

        C1_per_array.append(per_C(C1, mu_ch3oh))
        C2_per_array.append(per_C(C2, mu_hcho))
        C3_per_array.append(per_C(C3, mu_o2))
        C4_per_array.append(per_C(C4, mu_h2))
        t_array.append(t)

        # проверка стационарности в процентах
        if (abs(C1_per_array[-1] - C1_per_array[-2]) < eps and
            abs(C2_per_array[-1] - C2_per_array[-2]) < eps and
            abs(C3_per_array[-1] - C3_per_array[-2]) < eps and
            abs(C4_per_array[-1] - C4_per_array[-2]) < eps):
            break

    result = {
        "t": np.array(t_array),
        "C1": np.array(C1_array),
        "C2": np.array(C2_array),
        "C3": np.array(C3_array),
        "C4": np.array(C4_array),
        "C1_percent": np.array(C1_per_array),
        "C2_percent": np.array(C2_per_array),
        "C3_percent": np.array(C3_per_array),
        "C4_percent": np.array(C4_per_array),
        "C2_out_mol": C2_array[-1],
        "C2_out_percent": C2_per_array[-1],
        "t_final": t_array[-1]
    }
    return result


# --- Имитационное моделирование выходная координата СНСНОвых, входная координата m_CH3OHвх ---

if __name__ == "__main__":
    V_optim = 5.0
    T_optim = 950.0

    C2_out_array = [0]
    C1_in_array = []
    t_out_array = [0]
    print(m_c1_vh)
    for m in m_c1_vh:
        res = solve_model(V_optim, T_optim, m)
        C1_in_array.append(res['C1'][0])
        C2_out_array.append(res['C2_out_mol'])
        t_out_array.append(t_out_array[-1]+res['t_final'])

    print(f"V = {V_optim} м³, T = {T_optim} K")

    # графики:
    plt.figure(figsize=(15, 5))

    plt.subplot(3, 1, 1)
    plt.plot(m_c1_vh)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('m (CH3OH)вх')

    plt.subplot(3, 1, 2)
    plt.plot(C1_in_array)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C1 (CH3OH)вх')

    plt.subplot(3, 1, 3)
    plt.plot(C2_out_array)
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C2 (HCHO)вых')

    plt.tight_layout()
    plt.show()
