import numba
from functions import distance
from functions import focus_on


@numba.njit(cache=True)
def molecule_potential_energy(i, molecules, L):  # i — номер молекулы, molecules — набор молекул в заданном состоянии
    #   Arguments
    #   i [int] выделенная молекула
    #   molecules [float64, float64] двумерный массив с молекулами в заданном состоянии
    #   L [double] длина ребра ячейки моделирования
    focused_molecules = focus_on(i, molecules, L)  # фокусируемся на выделенной молекуле (мысленно помещаем её в центр)
    # это необходимо для учёта взаимодействия с частицами из соседних ячеек
    U = 0
    for j in range(len(molecules)):
        if j != i:
            U += potential_energy(distance(focused_molecules[i], focused_molecules[j]))
            # суммируем потенциальную энергию по всем молекулам
    return U


@numba.njit(cache=True)  # потенциал Леннарда-Джонса
def potential_energy(r):
    return 4 * (r ** (-12) - r ** (-6))
