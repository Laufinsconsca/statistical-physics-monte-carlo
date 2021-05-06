import numba
import numpy as np
from functions import shift, output_execution_progress

from molecule_potential_energy import molecule_potential_energy


@numba.njit(cache=True)
def play_out_conditions(molecules_ensemble, M, N, delta, L, T, execution_progress_struct):
    """
    Функция разыгрывания состояний

    :param molecules_ensemble: массив, содержащий набор частиц в начальном состоянии
    :param M: общее число состояний
    :param N: общее число частиц
    :param delta: параметр сдвига
    :param L: длина ребра ячейки моделирования
    :param T: безразмерная температура
    :param execution_progress_struct: класс типа ExecutionProgress, хранящий параметры вывода процента выполнения
    в консоль
    :return: массив, содержащий набор частиц во всех учитываемых (неотсянных) состояниях
    """
    progress = 0
    p = 1  # период отображения процента выполнения (в итерациях)
    M_accounted = len(molecules_ensemble)  # количество учитываемых состояний
    h_p = 100 / M
    if execution_progress_struct.output_progress_to_console:
        while execution_progress_struct.lower_bound > h_p * p:
            p += 1
    for m in range(M - 1):  # идём в цикле от 1 до последнего состояния
        prev = molecules_ensemble[m % M_accounted]  # определяем предыдущее состояние как prev
        num = np.random.randint(0, N)  # разыграли номер молекулы
        coordinate = np.random.randint(0, 3)  # разыграли координату
        shift_ = delta * np.random.uniform(-1, 1)  # разыграли сдвиг по координате
        U_previous = molecule_potential_energy(prev, num, L)
        # считаем потенциальную энергию выбранной частицы в предыдущем состоянии
        cur = shift(prev, shift_, num, coordinate, L)  # определяем новый набор частиц с учётом сдвига
        U_current = molecule_potential_energy(cur, num, L)
        # считаем потенциальную энергию выбранной частицы в текущем состоянии
        diff = U_current - U_previous  # разница энергий
        if not (diff < 0 or np.random.uniform(0, 1) < np.exp(-diff / T)):  # проверяем условие принятия состояния
            cur = prev  # возвращаем частицу на место если новое состояние не подошло
        molecules_ensemble[0 if m % M_accounted == M_accounted - 1 else (m % M_accounted) + 1] = cur
        # сохраняем с перезаписыванием новое состояние в ансамбль
        if execution_progress_struct.output_progress_to_console:
            progress += h_p
            if m % p == 0:
                output_execution_progress(execution_progress_struct, "Разыгрывание состояний",
                                          progress)
    return molecules_ensemble
