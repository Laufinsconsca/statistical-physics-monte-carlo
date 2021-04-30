import numpy as np
import numba


@numba.njit(cache=True)
def populate_cube(n, L):
    #   Arguments
    #   n [int] число ячеек вдоль одной координаты при остальных двух фиксированных
    #   L [double] длина ребра ячейки
    molecules = np.zeros((n ** 3, 3))  # пустой куб
    for i in range(n):
        for j in range(n):
            for k in range(n):
                molecules[i * n * n + j * n + k][0] = L * (i - 1) + L/2
                molecules[i * n * n + j * n + k][1] = L * (j - 1) + L/2
                molecules[i * n * n + j * n + k][2] = L * (k - 1) + L/2
                # заполяем куб заданным количеством частиц, помещая каждую частицу в центр своей ячейки
                # распределение частиц
    return molecules
