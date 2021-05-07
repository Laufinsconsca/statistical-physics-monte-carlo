import numba
from numpy import sqrt


@numba.njit(cache=True)
def molecule_potential_energy(molecules, num, L):
    """
    Потенциальная энергия выделенной молекулы

    :param molecules: массив молекул в некотором состоянии
    :param num: номер выделенной молекулы
    :param L: длина ребра ячейки моделирования
    :return: безразмерная потенциальная энергия молекулы
    """
    U = 0
    d = 0
    x_shift = L / 2 - molecules[num][0]
    y_shift = L / 2 - molecules[num][1]
    z_shift = L / 2 - molecules[num][2]
    for i in range(len(molecules)):
        if num != i:
            x = molecules[i][0]
            y = molecules[i][1]
            z = molecules[i][2]
            if x + x_shift >= L:
                temp = x + x_shift - L
            elif x + x_shift < 0:
                temp = x + x_shift + L
            else:
                temp = x + x_shift
            d += (temp - L / 2) ** 2
            if y + y_shift >= L:
                temp = y + y_shift - L
            elif y + y_shift < 0:
                temp = y + y_shift + L
            else:
                temp = y + y_shift
            d += (temp - L / 2) ** 2
            if z + z_shift >= L:
                temp = z + z_shift - L
            elif z + z_shift < 0:
                temp = z + z_shift + L
            else:
                temp = z + z_shift
            d += (temp - L / 2) ** 2
            U += potential_energy(sqrt(d))
            d = 0
    return U


@numba.njit(cache=True)
def potential_energy(r):
    """
    Потенциал Леннарда-Джонса

    :param r: аргумент функции (безразмерное расстояние)
    :return: безразмерная потенциальная энергия
    """
    return 4 * (r ** (-12) - r ** (-6))
