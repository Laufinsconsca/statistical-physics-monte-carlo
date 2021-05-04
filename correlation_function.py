import numba
import numpy as np
from numba import prange

from functions import distance, output_execution_percentage
from functions import focus_on_given_molecule


@numba.njit(cache=True, parallel=True)
def calculate_pair_correlation_function(molecules_ensemble, r, delta_r, L, n, execution_percentage_struct):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_percentage_struct: класс типа ExecutionPercentage, хранящий параметры отображения процента выполнения
    :return: массив значений корреляционной функции
    """
    pair_corr_func = np.zeros(len(r))
    if execution_percentage_struct.output_percentage_to_console:
        percentage_of_completion = 0
        h_p = 100 / len(r)
        p = 1  # период отображения процента выполнения (в итерациях)
        while execution_percentage_struct.lower_bound > h_p * p:
            p += 1
        for i in range(len(r)):
            delta_N = 0
            for j in prange(len(molecules_ensemble)):
                for k in range(len(molecules_ensemble[0])):
                    delta_N += number_of_molecules_in_a_spherical_layer(molecules_ensemble[j], r[i], delta_r, k, L)
            percentage_of_completion += h_p
            if i % p == 0:
                output_execution_percentage(execution_percentage_struct, "Расчёт корреляционной функции",
                                            percentage_of_completion)
            delta_N = delta_N / (len(molecules_ensemble) * len(molecules_ensemble[0]))
            # среднее количество частиц в шаровом слое
            pair_corr_func[i] = delta_N / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
    else:
        for i in prange(len(r)):
            delta_N = 0
            for j in range(len(molecules_ensemble)):
                for k in range(len(molecules_ensemble[0])):
                    delta_N += number_of_molecules_in_a_spherical_layer(molecules_ensemble[j], r[i], delta_r, k, L)
            delta_N = delta_N / (len(molecules_ensemble) * len(molecules_ensemble[0]))
            # среднее количество частиц в шаровом слое
            pair_corr_func[i] = delta_N / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
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
    :return: количество молекул в сферическом слое выделенной молекулы
    """
    focused_molecules = focus_on_given_molecule(molecules, num, L)
    # фокусируемся на выделенной молекуле (мысленно помещаем её в центр ячейки моделирования)
    # это необходимо для учёта частиц из соседних ячеек
    N = 0
    for i in range(len(molecules)):
        if i != num and (r - delta_r / 2) < distance(focused_molecules[num], focused_molecules[i]) < (r + delta_r / 2):
            N += 1
    return N
