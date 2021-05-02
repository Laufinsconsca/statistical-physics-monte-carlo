import numpy as np
import numba
from numba import prange
from functions import distance
from functions import focus_on_given_molecule


@numba.njit(cache=True, parallel=True)
def calculate_pair_correlation_function(molecules_ensemble, r, delta_r, L, n):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :return: массив значений корреляционной функции
    """
    pair_corr_func = np.zeros((len(r)))
    for j in prange(len(r)):
        delta_N = 0
        for k in range(len(molecules_ensemble)):
            for i in range(len(molecules_ensemble[0])):
                delta_N += number_of_molecules_in_a_spherical_layer(molecules_ensemble[k], r[j], delta_r, i, L)
        delta_N = delta_N / (len(molecules_ensemble) * len(molecules_ensemble[0]))
        # среднее количество частиц в шаровом слое
        pair_corr_func[j] = delta_N / (4 * np.pi * r[j] * r[j] * delta_r * n)  # корреляционная функция
    return pair_corr_func


@numba.njit(cache=True)
def number_of_molecules_in_a_spherical_layer(molecules, r, delta_r, num, L):
    """
    Вычисление количества молекул в сферическом слое выделенной молекулы

    :param molecules: массив молекул в некотором состоянии
    :param r: расстояние, на которое отстоит середина толщины сферического слоя от выделенной молекулы
    :param delta_r: толщина сферического слоя
    :param num: номер выделенной молекулы
    :param L: длина ребра ячейки моделирования
    :return: количество молекул в сферическом слое
    """
    focused_molecules = focus_on_given_molecule(molecules, num, L)
    # фокусируемся на выделенной молекуле (мысленно помещаем её в центр) ячейки моделирования
    # это необходимо для учёта частиц из соседних ячеек
    N = 0
    for j in range(len(molecules)):
        if j != num:
            N += is_there_the_molecule_in_the_spherical_layer(focused_molecules[num], focused_molecules[j], r, delta_r)
    return N


@numba.njit(cache=True)
def is_there_the_molecule_in_the_spherical_layer(given_molecule, a_molecule, r, delta_r):
    """
    Проверка попала ли частица в шаровой слой

    :param given_molecule: молекула, относительно которой берётся сферический слой
    :param a_molecule: любая молекула
    :param r: расстояние, на которое отстоит середина толщины сферического слоя от выделенной молекулы
    :param delta_r: толщина сферического слоя
    :return: 1 если молекула попала в сферический слой, 0 – иначе
    """
    if (r - delta_r / 2) < distance(given_molecule, a_molecule) < (r + delta_r / 2):
        return 1
    return 0
