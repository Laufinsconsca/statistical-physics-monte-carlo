import numba
from functions import distance
from functions import focus_on_given_molecule


@numba.njit(cache=True)
def molecule_potential_energy(molecules, num, L):
    """
    Потенциальная энергия выделенной молекулы

    :param molecules: массив молекул в некотором состоянии
    :param num: номер выделенной молекулы
    :param L: длина ребра ячейки моделирования
    :return: безразмерная потенциальная энергия молекулы
    """
    focused_molecules = focus_on_given_molecule(molecules, num, L)
    # фокусируемся на выделенной молекуле (мысленно помещаем её в центр)
    # это необходимо для учёта взаимодействия с частицами из соседних ячеек
    U = 0
    for j in range(len(molecules)):
        if j != num:
            U += potential_energy(distance(focused_molecules[num], focused_molecules[j]))
            # суммируем потенциальную энергию по всем молекулам, кроме заданной
    return U


@numba.njit(cache=True)
def potential_energy(r):
    """
    Потенциал Леннарда-Джонса

    :param r: аргумент функции (безразмерное расстояние)
    :return: безразмерная потенциальная энергия
    """
    return 4 * (r ** (-12) - r ** (-6))
