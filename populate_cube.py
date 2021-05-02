import numpy as np
import numba


@numba.njit(cache=True)
def populate_cube(n, L):
    """
    Заполнение куба частицами (начальное состояние)

    :param n: число ячеек вдоль одной координаты при остальных двух фиксированных
    :param L: длина ребра ячейки моделирования, содержащей одну частицу
    :return: куб, определенный образом заполненный частицами
    """
    molecules = np.zeros((n ** 3, 3))  # инициализируем пустой куб
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # заполяем куб заданным количеством частиц, помещая каждую частицу в центр своей ячейки
                molecules[i * n * n + j * n + k][0] = L * (i - 1) + L/2
                molecules[i * n * n + j * n + k][1] = L * (j - 1) + L/2
                molecules[i * n * n + j * n + k][2] = L * (k - 1) + L/2
    return molecules
