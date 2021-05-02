import matplotlib.pyplot as plt  # отрисовка графиков
import numpy as np
import numba


def plot(x, f, title, x_label, y_label):
    """
    Построение графика

    :param x: массив аргументов функции
    :param f: массив значений функции
    :param title: заглавие графика
    :param x_label: обозначение шкалы абсцисс
    :param y_label: обозначение шкалы ординат
    """
    plt.figure(figsize=(11, 8), dpi=80)
    # plt.scatter(x, f, s=5) # построение графика в виде набора точек
    plt.plot(x, f)  # построение сплошного графика
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


@numba.njit(cache=True)
def distance(first, second):
    """
    Расстояние между двумя молекулами

    :param first: первая молекула
    :param second: вторая молекула
    :return: расстояние между двумя молекулами
    """
    return np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2)


@numba.njit(cache=True)
def shift(molecules, shift_, num, coordinate, L):
    """
    Смещение в массиве молекул одной заданной молекулы в заданном направлении на заданную величину

    :param molecules: массив молекул в нектором состоянии
    :param shift_: величина смещения
    :param num: номер заданной молекулы
    :param coordinate: номер заданной координаты (0 – "x", 1 – "y", 2 – "z")
    :param L: длина ребра ячейки моделирования
    :return: массив частиц в некотором состоянии, в котором одна из частиц подвеглась сдвигу
    """
    shifted_molecules = molecules.copy()
    if coordinate == 0:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 0, L)
    elif coordinate == 1:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 1, L)
    elif coordinate == 2:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 2, L)
    return shifted_molecules


@numba.njit(cache=True)
def shift_a_molecule(molecule, shift_, coordinate, L):  # сдвиг молекулы с учётом выхода за границу ячейки моделирования
    """
    Сдвиг молекулы

    :param molecule: некоторая молекула
    :param shift_: величина сдвига
    :param coordinate: номер заданной координаты (0 – "x", 1 – "y", 2 – "z")
    :param L: длина ребра ячейки моделирования
    :return: молекула, сдвинутая на заданную величину в заданном направлении
    """

    if molecule[coordinate] + shift_ >= L:
        molecule[coordinate] = molecule[coordinate] + shift_ - L
    elif molecule[coordinate] + shift_ < 0:
        molecule[coordinate] = molecule[coordinate] + shift_ + L
    else:
        molecule[coordinate] = molecule[coordinate] + shift_
    return molecule


@numba.njit(cache=True)
def focus_on_given_molecule(molecules, num, L):
    """
    Фокус на выделенной молекуле (помещение её в центр ячейки моделирования).
    Это необходимо для учёта взаимодействия с частицами из соседних ячеек

    :param molecules: массив молекул в некотором состоянии
    :param num: номер выделенной молекулы
    :param L: длина ребра ячейки моделирования
    :return: массив молекул в некотором состоянии, содержащий выделенную частицу в центре ячейки моделирования
    """
    focused_molecules = np.zeros_like(molecules)
    # создаём заполненный нулями массив той же размерности, что и molecules
    x_shift = L / 2 - molecules[num][0]
    y_shift = L / 2 - molecules[num][1]
    z_shift = L / 2 - molecules[num][2]
    for k in range(len(molecules)):
        x = molecules[k][0]
        y = molecules[k][1]
        z = molecules[k][2]
        if x + x_shift >= L:
            focused_molecules[k][0] = x + x_shift - L
        elif x + x_shift < 0:
            focused_molecules[k][0] = x + x_shift + L
        else:
            focused_molecules[k][0] = x + x_shift
        if y + y_shift >= L:
            focused_molecules[k][1] = y + y_shift - L
        elif y + y_shift < 0:
            focused_molecules[k][1] = y + y_shift + L
        else:
            focused_molecules[k][1] = y + y_shift
        if z + z_shift >= L:
            focused_molecules[k][2] = z + z_shift - L
        elif z + z_shift < 0:
            focused_molecules[k][2] = z + z_shift + L
        else:
            focused_molecules[k][2] = z + z_shift
    return focused_molecules
