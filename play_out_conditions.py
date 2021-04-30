import numba
import numpy as np
from functions import shift

from molecule_potential_energy import molecule_potential_energy


@numba.njit(cache=True)
def play_out_conditions(M, molecules_ensemble, N, delta, L, T):
    for m in range(M - 1):  # идём в цикле от 2 до последнего состояния
        num = np.random.randint(0, N)  # разыграли номер молекулы
        prev = molecules_ensemble[m]  # определяем предыдущее состояние как prev
        coordinate = np.random.randint(0, 3)  # разыграли координату
        shift_ = delta * np.random.uniform(-1, 1)  # разыграли сдвиг по координате
        U_previous = molecule_potential_energy(num, prev,
                                               L)  # считаем потенциальную энергию выбранной частицы в предыдущем
        # состоянии
        cur = shift(prev, shift_, num, coordinate, L)  # определяем новый набор частиц с учётом сдвига
        U_current = molecule_potential_energy(num, cur,
                                              L)  # считаем потенциальную энергию выбранной частицы в текущем состоянии
        diff = U_current - U_previous  # разница энергий
        if not (diff < 0 or np.random.uniform(0, 1) < np.exp(-diff / T)):  # проверяем условие принятия состояния
            cur = prev  # возвращаем частицу на место если новое состояние не подошло
        molecules_ensemble[m + 1] = cur  # сохраняем новое состояние в ансамбль
    return molecules_ensemble
