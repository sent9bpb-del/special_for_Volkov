import numpy as np  # Импортируем библиотеку для численных вычислений и работы с массивами
import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков
from math_litovka import hcho_objective, pozitive_hcho_objective  # Импортируем функции из модуля для математической модели

# ---- ТЕСТОВАЯ ФУНКЦИЯ ----
# f(x1, x2) = x1^2 + x2^2
def test_f(x1, x2):
    return x1**2 + x2**2  # Возвращает сумму квадратов x1 и x2

# ---- РИСОВАНИЕ ЛИНИЙ УРОВНЯ И ТРАЕКТОРИИ ----
def draw_contour_cd(path_x, path_y, f0, bounds, grid_n=30):
    """
    Функция для рисования контурных линий и траектории поиска минимума
    path_x, path_y — координаты траектории,
    f0 — целевая функция,
    bounds — границы для координат (x1_min, x1_max), (x2_min, x2_max),
    grid_n — количество точек на сетке для контурного графика.
    """
    x1_min, x1_max = bounds[0]  # Границы по x1
    x2_min, x2_max = bounds[1]  # Границы по x2

    # Генерация сетки значений для x1 и x2 в указанных границах
    x1 = np.linspace(x1_min - 10, x1_max + 10, grid_n)  # Расстояние для x1
    x2 = np.linspace(x2_min - 1, x2_max + 1, grid_n)    # Расстояние для x2
    X1, X2 = np.meshgrid(x1, x2)  # Создаём сетку для контурного графика

    # Считаем значения целевой функции f0 в каждой точке сетки
    Z = np.zeros_like(X1, dtype=float)  # Инициализация массива для значений функции
    for i in range(grid_n):
        for j in range(grid_n):
            T_val = X1[j, i]  # Текущая точка по x1
            V_val = X2[j, i]  # Текущая точка по x2
            Z[j, i] = f0(T_val, V_val)  # Вычисляем значение функции в этой точке

    # Настройка и вывод контурного графика
    plt.figure(figsize=(6, 5))  # Устанавливаем размер графика
    cs = plt.contour(X1, X2, Z, levels=20)  # Рисуем контуры с 20 уровнями
    plt.clabel(cs, inline=True, fontsize=8)  # Добавляем метки для контуров

    # Рисуем траекторию поиска минимума
    plt.plot(path_x, path_y, 'o-', linewidth=1.5)  # Траектория поиска (маркер и линии)

    # Добавляем точки старта и финиша
    plt.scatter(path_x[0], path_y[0],
                color='lime', s=80, edgecolor='black',
                zorder=3, label='start')  # Стартовая точка
    plt.scatter(path_x[-1], path_y[-1],
                color='red', s=80, edgecolor='black',
                zorder=3, label='finish')  # Конечная точка

    plt.xlabel('x1')  # Подпись оси X
    plt.ylabel('x2')  # Подпись оси Y
    plt.title('Покоординатный спуск')  # Заголовок графика
    plt.grid(True)  # Включаем сетку
    plt.legend()  # Включаем легенду
    plt.show()  # Отображаем график

# ---- ПОКООРДИНАТНЫЙ СПУСК С УТОЧНЕНИЕМ ШАГА ----
def coordinate_descent(x1, x2, f0, bounds,
                       eps=1e-4, max_iter=50,
                       passes=4, steps=25):
    """
    Реализация метода покоординатного спуска для поиска минимума.
    x1, x2  – начальная точка,
    f0      – функция, которую минимизируем,
    bounds  – ограничения для x1 и x2 ((x1_min, x1_max), (x2_min, x2_max)),
    eps     – точность по норме вектора (x1, x2),
    passes  – количество уточнений интервала вдоль координаты,
    steps   – количество шагов в сетке для каждой координаты.
    """

    x1_min, x1_max = bounds[0]  # Границы для x1
    x2_min, x2_max = bounds[1]  # Границы для x2

    path_x = [x1]  # Список для хранения значений x1 на пути
    path_y = [x2]  # Список для хранения значений x2 на пути

    # Локальный поиск минимума по одной координате
    def search_coord(coord_idx, left, right, x1, x2):
        """
        Поиск минимума по одной координате.
        coord_idx – индекс координаты (0 для x1, 1 для x2),
        left, right – границы для поиска,
        x1, x2 – текущие значения переменных.
        """
        base_left, base_right = left, right  # Изначальные границы
        best_coord = None  # Лучшая найденная координата
        best_val = None  # Лучшее значение функции

        for _ in range(passes):  # Повторяем поиск несколько раз для уточнения
            grid = np.linspace(left, right, steps)  # Создаём сетку точек

            # Ищем минимум по каждой точке сетки
            for val in grid:
                if coord_idx == 0:
                    val_f = f0(val, x2)  # Вычисляем значение функции по x1
                else:
                    val_f = f0(x1, val)  # Вычисляем значение функции по x2

                if best_val is None or val_f < best_val:  # Если нашли лучшее значение
                    best_val = val_f
                    best_coord = val

            # Сужаем интервал вокруг найденной лучшей точки
            delta = (right - left) / (steps - 1)
            left = max(base_left, best_coord - delta)
            right = min(base_right, best_coord + delta)

        # Обновляем значения переменных в зависимости от координаты
        if coord_idx == 0:
            x1_new, x2_new = best_coord, x2  # Обновляем x1
        else:
            x1_new, x2_new = x1, best_coord  # Обновляем x2

        return x1_new, x2_new, best_val  # Возвращаем обновленные значения и минимум

    iters = 0  # Счётчик итераций

    for _ in range(max_iter):  # Максимальное количество итераций
        x1_old, x2_old = x1, x2  # Запоминаем старые значения

        # Шаг по x1
        x1, x2, _ = search_coord(0, x1_min, x1_max, x1, x2)
        path_x.append(x1)  # Добавляем новое значение x1 в путь
        path_y.append(x2)  # Добавляем новое значение x2 в путь

        # Шаг по x2
        x1, x2, _ = search_coord(1, x2_min, x2_max, x1, x2)
        path_x.append(x1)  # Добавляем новое значение x1 в путь
        path_y.append(x2)  # Добавляем новое значение x2 в путь

        iters += 1  # Увеличиваем количество итераций

        # Критерий остановки: если изменения в x1 и x2 меньше заданной точности
        if np.hypot(x1 - x1_old, x2 - x2_old) < eps:
            break

    return x1, x2, iters, path_x, path_y  # Возвращаем результат поиска минимума

# ---- ТЕСТ НА ФУНКЦИИ x1^2 + x2^2 ----
bounds = ((-10, 10), (-10, 10))  # Границы для x1 и x2

print("Решение для тестовой функции f(x1, x2) = x1^2 + x2^2:")
x1, x2, iters, path_x, path_y = coordinate_descent(-1.3, -7, test_f, bounds)  # Запуск метода покоординатного спуска

# Выводим результаты
print(
    f"X* = ({x1:.4f}, {x2:.4f}); "
    f"f(X*) = {test_f(x1, x2):.4f}, "
    f"число итераций = {iters}"
)

# Рисуем контуры и траекторию
draw_contour_cd(path_x, path_y, test_f, bounds)

# ---- РЕШЕНИЕ ДЛЯ МАТЕМАТИЧЕСКОЙ МОДЕЛИ ----
print("\nРешение для математической модели: ")
bounds = ((800, 950), (2, 5))  # Границы для T и V
T0, V0 = 830.0, 2.5  # Начальные значения T и V
x1, x2, iters, path_x, path_y = coordinate_descent(T0, V0, hcho_objective, bounds)  # Запуск метода покоординатного спуска

# Выводим результаты
print(
    f"X* = ({x1:.4f}, {x2:.4f}); "
    f"f(X*) = {pozitive_hcho_objective(x1, x2):.4f}, "
    f"число итераций = {iters}"
)

# Рисуем контуры и траекторию
draw_contour_cd(path_x, path_y, pozitive_hcho_objective, bounds)
