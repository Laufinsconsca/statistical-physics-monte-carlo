import numpy as np
import numba
from numba import prange
from functions import distance
from functions import focus_on


@numba.njit(cache=True, parallel=True)
def calculate_correlation_function(r, M, molecules_ensemble, M_relax, delta_r, L, n):
    corr_func = np.zeros((len(r)))
    for j in prange(len(r)):
        delta_N = 0
        k = M_relax
        while k < M:
            for i in range(len(molecules_ensemble[0])):
                delta_N += number_of_molecules_in_a_spherical_layer(r[j], molecules_ensemble[k], i, delta_r, L)
            k += 1
        delta_N = delta_N / ((M - M_relax) * len(molecules_ensemble[0]))
        # среднее количество частиц в шаровом слое
        corr_func[j] = delta_N / (4 * np.pi * r[j] * r[j] * delta_r * n)  # корреляционная функция
    return corr_func


@numba.njit(cache=True)
def number_of_molecules_in_a_spherical_layer(r, molecules, i, delta_r, L):
    #  Arguments
    #  r [double] аргумент корреляционной функции
    #  i [int] номер молекулы относительно которой рассчитывается парная корреляционная функция
    focused_molecules = focus_on(i, molecules, L)
    # фокусируемся на выделенной молекуле (мысленно помещаем её в центр) ячейки моделирования
    # это необходимо для учёта частиц из соседних ячеек
    N = 0
    for j in range(len(molecules)):
        if j != i:
            N += is_there_the_molecule_in_the_spherical_layer(focused_molecules[i], focused_molecules[j], r, delta_r)
    return N


@numba.njit(cache=True)
def is_there_the_molecule_in_the_spherical_layer(given_molecule, a_molecule, r, delta_r):
    if (r - delta_r / 2) < distance(given_molecule, a_molecule) < (r + delta_r / 2):
        # проверка попала ли частица в шаровой слой
        return 1
    return 0
