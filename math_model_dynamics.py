import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков
import numpy as np               # Импортируем библиотеку для численных вычислений и работы с массивами

# --- Константы и параметры модели ---
R = 8.31        # Универсальная газовая постоянная, Дж/(моль·К)
A1 = 50         # Предэкспоненциальный множитель для первой реакции
E1 = 64000      # Энергия активации первой реакции, Дж/моль
A2 = 10         # Предэкспоненциальный множитель для второй реакции
E2 = 44000      # Энергия активации второй реакции, Дж/моль
A3 = 10**6      # Предэкспоненциальный множитель для третьей реакции
E3 = 108000     # Энергия активации третьей реакции, Дж/моль
A4 = 25 * 10**5 # Предэкспоненциальный множитель для четвертой реакции
E4 = 112000     # Энергия активации четвертой реакции, Дж/моль

# Массовые расходы веществ на входе, кг/с
m_c1_vh = 0.28      # Массовый расход метанола
m_vozd_vh = 0.28    # Массовый расход воздуха

ro = 1.6            # Плотность газовой смеси, кг/м^3

# Молекулярные массы веществ, кг/моль
mu_ch3oh = 0.032    # Молекулярная масса метанола CH3OH
mu_o2 = 0.032      # Молекулярная масса кислорода O2
mu_hcho = 0.030    # Молекулярная масса формальдегида HCHO
mu_h2 = 0.002      # Молекулярная масса водорода H2

# Суммарный массовый расход, кг/с
m = m_c1_vh + m_vozd_vh

# --- Функции для вычислений ---

def k(A, E, T):
    """
    Функция для вычисления константы скорости реакции по уравнению Аррениуса.
    A - предэкспоненциальный множитель,
    E - энергия активации,
    T - температура в Кельвинах.
    """
    return A * np.exp(-E / (R * T))  # Расчёт по формуле Аррениуса

def per_C(C_mol_per_m3, mu):
    """
    Функция для перевода концентрации вещества из моль/м^3 в массовую долю в процентах.
    C_mol_per_m3 - концентрация в моль/м^3,
    mu - молекулярная масса вещества, кг/моль.
    """
    return (C_mol_per_m3 * 100 * mu) / ro  # Перевод в проценты по массе

def solve_model(V, T, d_t=0.01, eps=0.01, max_steps=200000):
    """
    Основная функция для решения динамической модели.
    Входные параметры:
    - V - объем реактора (м^3)
    - T - температура (К)
    - d_t - шаг по времени (с)
    - eps - точность для проверки стационарности
    - max_steps - максимальное количество шагов интегрирования
    """
    # Входные концентрации (моль/м^3)
    C1_vh = (m_c1_vh * ro) / (m * mu_ch3oh)          # Концентрация метанола на входе
    C3_vh = (m_vozd_vh * 0.22 * ro) / (m * mu_o2)    # Концентрация кислорода на входе (22% O2)
    C2_vh = 0.0                                      # Концентрация формальдегида на входе
    C4_vh = 0.0                                      # Концентрация водорода на входе

    # Начальные концентрации на входе в аппарат
    C1 = C1_vh
    C2 = C2_vh
    C3 = C3_vh
    C4 = C4_vh
    t = 0.0  # Начальное время

    # Гидродинамика
    v = m / ro          # Объемный расход (м^3/с)
    tau = V / v         # Время пребывания в реакторе (с)

    # Константы скоростей реакций при заданной температуре T
    k1 = k(A1, E1, T)
    k2 = k(A2, E2, T)
    k3 = k(A3, E3, T)
    k4 = k(A4, E4, T)

    # Массивы для хранения значений
    C1_array = [C1]
    C2_array = [C2]
    C3_array = [C3]
    C4_array = [C4]

    C1_per_array = [per_C(C1, mu_ch3oh)]
    C2_per_array = [per_C(C2, mu_hcho)]
    C3_per_array = [per_C(C3, mu_o2)]
    C4_per_array = [per_C(C4, mu_h2)]
    t_array = [t]

    # Интегрирование до стационарного состояния
    for step in range(max_steps):
        # Расчёт производных концентраций (dC/dt)
        dC1dt = (1 / tau) * (C1_vh - C1) - 2 * k1 * C1 * C3 - k2 * C1 - 2 * k3 * C1 * C3 - k4 * C1 * C4
        dC2dt = (1 / tau) * (C2_vh - C2) + 2 * k1 * C1 * C3 + k2 * C1
        dC3dt = (1 / tau) * (C3_vh - C3) - k1 * C1 * C3 - 3 * k3 * C1 * C3
        dC4dt = (1 / tau) * (C4_vh - C4) - k4 * C1 * C4 + k2 * C1

        # Шаг Эйлера для интегрирования
        C1 += dC1dt * d_t
        C2 += dC2dt * d_t
        C3 += dC3dt * d_t
        C4 += dC4dt * d_t
        t += d_t  # Увеличиваем время

        # Записываем новые значения в массивы
        C1_array.append(C1)
        C2_array.append(C2)
        C3_array.append(C3)
        C4_array.append(C4)

        C1_per_array.append(per_C(C1, mu_ch3oh))
        C2_per_array.append(per_C(C2, mu_hcho))
        C3_per_array.append(per_C(C3, mu_o2))
        C4_per_array.append(per_C(C4, mu_h2))
        t_array.append(t)

        # Проверка на стационарность: если разница в концентрациях меньше eps, то выходим из цикла
        if (abs(C1_per_array[-1] - C1_per_array[-2]) < eps and
            abs(C2_per_array[-1] - C2_per_array[-2]) < eps and
            abs(C3_per_array[-1] - C3_per_array[-2]) < eps and
            abs(C4_per_array[-1] - C4_per_array[-2]) < eps):
            break

    # Результаты решения модели
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

# --- Целевая функция для оптимизации ---

def hcho_objective(T, V):
    """
    Целевая функция для оптимизации:
    по T и V считаем модель и возвращаем минус
    концентрации формальдегида на выходе в %.
    """
    res = solve_model(V, T)
    return -res["C2_out_percent"]

def pozitive_hcho_objective(T, V):
    """
    Целевая функция для оптимизации:
    по T и V считаем модель и возвращаем минус
    концентрации формальдегида на выходе в %.
    """
    res = solve_model(V, T)
    return res["C2_out_percent"]

# --- Пример одиночного прогона модели (для проверки) ---

if __name__ == "__main__":
    V_test = 3.0  # Объем реактора для теста
    T_test = 900.0  # Температура для теста

    res = solve_model(V_test, T_test)  # Запускаем модель с заданными параметрами

    # Выводим результаты
    print(f"V = {V_test} м³, T = {T_test} K")
    print(f"Концентрация метанола(CH3OH) на входе: "
          f"{res['C1'][0]:.4f} моль/м³, {res['C1_percent'][0]:.4f} %")
    print(f"Концентрация кислорода(O2) на входе: "
          f"{res['C3'][0]:.4f} моль/м³, {res['C3_percent'][0]:.4f} %")
    print(f"Концентрация формальдегида(HCHO) на выходе: "
          f"{res['C2_out_mol']:.4f} моль/м³, {res['C2_out_percent']:.4f} %")
    print(f"Время до стационара: {res['t_final']:.2f} c")

    # Построение графиков
    t = res["t"]
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(t, res["C1_percent"])
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C1 (CH3OH), %')

    plt.subplot(1, 3, 2)
    plt.plot(t, res["C3_percent"])
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C3 (O2), %')

    plt.subplot(1, 3, 3)
    plt.plot(t, res["C2_percent"])
    plt.grid(True)
    plt.xlabel('$t, c$')
    plt.ylabel('C2 (HCHO), %')

    plt.tight_layout()
    plt.show()
