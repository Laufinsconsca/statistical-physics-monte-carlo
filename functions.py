import matplotlib.pyplot as plt  # отрисовка графиков
import numpy as np
import numba


def plot(x, function, title, x_label, y_label):
    plt.figure(figsize=(11, 8), dpi=80)
    plt.scatter(x, function, s=5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()


@numba.njit(cache=True)
def distance(first, second):  # расстояние между двумя молекулами
    return np.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2)


@numba.njit(cache=True)
def shift(molecules, shift_, num, coordinate, L):  # сдвиг заданной молекулы в заданном направлении
    shifted_molecules = molecules.copy()
    if coordinate == 0:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 0, L)
    elif coordinate == 1:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 1, L)
    elif coordinate == 2:
        shifted_molecules[num] = shift_a_molecule(shifted_molecules[num], shift_, 2, L)
    return shifted_molecules


@numba.njit(cache=True)
def shift_a_molecule(molecule, shift_, coord, L):  # сдвиг молекулы с учётом выхода за границу ячейки моделирования
    if molecule[coord] + shift_ >= L:
        molecule[coord] = molecule[coord] + shift_ - L
    elif molecule[coord] + shift_ < 0:
        molecule[coord] = molecule[coord] + shift_ + L
    else:
        molecule[coord] = molecule[coord] + shift_
    return molecule


@numba.njit(cache=True)
def focus_on(i, molecules, L):  # фокусирование на выделенной частице (помещение её в центр ячейки моделирования)
    focused_molecules = np.zeros_like(molecules)
    # создаём заполненный нулями массив той же размерности что и molecules
    x_shift = L / 2 - molecules[i][0]
    y_shift = L / 2 - molecules[i][1]
    z_shift = L / 2 - molecules[i][2]
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
