import datetime
import time
from os import makedirs
from os import path

import numpy as np

from correlation_function import calculate_pair_correlation_function_on_cpu, calculate_pair_correlation_function_on_gpu
from execution_progress import ExecutionProgress
from functions import plot
from noble_gas import ChooseNobleGas
from play_out_states import play_out_states
from populate_cube import populate_cube
from validation import check_n_is_integer, check_M_relax_less_than_M
from enum import Enum


class CalculateOn(Enum):
    CPU = 0
    GPU = 1


if __name__ == '__main__':
    """
    МОДЕЛИРОВАНИЕ ПАРНОЙ КОРРЕЛЯЦИОННОЙ ФУНКЦИИ ДЛЯ БЛАГОРОДНЫХ ГАЗОВ В ЖИДКОМ СОСТОЯНИИ
    Краткое описание:
    Корреляционная функция показывает отношение концентрации частиц на данном расстоянии к средней концентрации частиц.
    Около нуля функция равна нулю, так как частицы не могут близко подойти друг к другу.
    При стремлении расстояния между частицами к бесконечности концентрация стремится к средней, 
    поэтому отношение стремится к единице.
    Для описания взаимодействия между частицами используется потенциал Леннарда-Джонса.
    """
    #  --------------------------- <общие константы> -------------------------------------------------------------------
    k = 1.38e-23  # постоянная Больцмана, Дж/К
    device_to_calculate_pair_correlation_function = CalculateOn.CPU
    # выбор устройства для расчёта корреляционной функции (если у вас несколько GPU, то можно указать номер устройства
    # вручную, установив в переменных средах целочисленное значение переменной CUDA_DEVICE,
    # по умолчанию выбирается первое устройство, поддерживающее CUDA)
    # учтите: расчёт на GPU поддерживается только видеокартами компании Nvidia
    # также для расчёта на GPU необходимо в переменных средах добавить путь к cl.exe в переменную Path,
    # пример расположения cl.exe:
    # C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64
    block_dim = (2, 2, 64)  # размерность блока (задаётся в случае вычисления на GPU)
    # максимальная размерность блока ограничена возможностями GPU (как каждая из размерностей, так и их произведение)
    #  -------------------------- </общие константы> -------------------------------------------------------------------
    #  -------------------------- <задаваемые константы задачи> --------------------------------------------------------
    N = 5 ** 3  # число частиц в одной ячейке (должно быть кубом натурального числа)
    M = 50000  # количество разыгрываемых состояний
    h_T = 0.1  # шаг изменения безразмерной температуры
    delta = 0.1  # константа рандомного сдвига молекулы
    M_relax = 49900  # количество отсеиваемых состояний (по умолчанию 1000)
    #  ----------------------------- <параметры газа> ------------------------------------------------------------------
    temperature = 85  # температура газа, K
    chosen_noble_gas = ChooseNobleGas.ARGON  # выбираем аргон
    #  ----------------------------- </параметры газа> -----------------------------------------------------------------
    #  ----------------------------- <параметры корреляционной функции> ------------------------------------------------
    h_r_left = 0.1  # шаг изменения аргумента корреляционной функции до первого пика
    h_r_the_first_peak_left = 0.01  # шаг изменения аргумента корреляционной функции вдоль первого пика слева от вершины
    h_r_top_of_the_first_peak = 0.001  # шаг изменения аргумента корреляционной функции на вершине первого пика
    h_r_the_first_peak_right = 0.01  # шаг изменения аргумента корреляционной функции вдоль первого пика справа
    # от вершины
    h_r_right = 0.01  # шаг изменения аргумента корреляционной функции после первого пика
    r_max_left = 0.8  # конец нуля корреляционной функции
    r_max_first_peak_left = 1.05  # конец левой части первого пика корреляционной функции
    r_max_top_of_the_first_peak = 1.15  # конец вершины пика корреляционной функции
    r_max_first_peak_right = 1.3  # конец правой части первого пика корреляционной функции
    delta_r = 0.3  # толщина шарового слоя (величина подобрана экспериментально)
    #  ----------------------------- </параметры корреляционной функции> -----------------------------------------------
    #  -------------------------- </задаваемые константы задачи> -------------------------------------------------------
    #  -------------------------- <валидация входных аргументов> -------------------------------------------------------
    N_coordinate = check_n_is_integer(N ** (1. / 3))  # количество частиц вдоль одной любой координаты
    # (с проверкой на целочисленность)
    check_M_relax_less_than_M(M, M_relax)
    #  -------------------------- </валидация входных аргументов> ------------------------------------------------------
    #  -------------------------- <отображение прогресса выполнения> ---------------------------------------------------
    #  ----------------------------- <отображение прогресса разыгрывания состояний> ------------------------------------
    output_play_out_progress_to_console = True  # позволяет отслеживать процент выполнения программы однако
    # снижает производительность (при N <= 1000 примерно на 10%)
    play_out_lower_bound_progress = 5
    # число, задающее нижнюю границу точности рассчитывания процента выполнения (в процентах)
    play_out_number_of_decimal_places = 2  # максимальное число дробных знаков
    play_out_description = "Разыгрывание состояний"
    #  ----------------------------- </отображение прогресса разыгрывания состояний> -----------------------------------
    #  ----------------------------- <отображение прогресса расчёта корреляционной функции> ----------------------------
    output_pair_corr_func_progress_to_console = True  # позволяет отслеживать процент выполнения программы однако
    # снижает производительность (при N <= 1000 примерно на 10%)
    pair_corr_func_lower_bound_progress = 1  # уменьшение параметра снижает производительность
    # так как генерируется больше выводов в консоль
    # число, задающее нижнюю границу точности расчитывания процента выполнения (в процентах)
    pair_corr_func_number_of_decimal_places = 3  # максимальное число дробных знаков
    pair_corr_func_description = "Вычисление парной корреляционной функции"
    #  ----------------------------- </отображение прогресса расчёта корреляционной функции> ---------------------------
    #  -------------------------- <отображение прогресса выполнения> ---------------------------------------------------
    #  -------------------------- <вычисляемые константы> --------------------------------------------------------------
    T0 = chosen_noble_gas.value.energy / k  # характерная температура задачи
    T_min = temperature / T0  # начальная температура расчёта
    T_max = temperature / T0  # конечная температура расчёта
    T = np.linspace(T_min, T_max, int((T_max - T_min) / h_T + 1))  # безразмерная температура
    n_concentration = chosen_noble_gas.value.ro / chosen_noble_gas.value.mass  # средняя концентрация
    n = n_concentration * (chosen_noble_gas.value.sigma ** 3)  # безразмерная концентрация
    # (sigma выступает в роли характерной длины задачи)
    V = N / n  # безразмерный объём
    L = V ** (1. / 3)  # безразмерная длина ребра ячейки моделирования
    #  ----------------------------- <параметры корреляционной функции> ------------------------------------------------
    r_min_left = delta_r + 0.01  # начальный аргумент корреляционной функции
    r_min_first_peak_left = r_max_left + h_r_the_first_peak_left  # начало левой части первого пика
    # корреляционной функции
    r_min_top_of_the_first_peak = r_max_first_peak_left + h_r_top_of_the_first_peak  # начало вершины пика
    # корреляционной функции
    r_min_first_peak_right = r_max_top_of_the_first_peak + h_r_the_first_peak_right  # начало правой части первого пика
    # корреляционной функции
    r_min_right = r_max_first_peak_right + h_r_right  # начало релаксации корреляционной функции
    r_max_right = (L - delta_r) / 2  # конечный аргумент корреляционной функции (расстояние от центра до грани куба)
    r_left = np.linspace(r_min_left, r_max_left, int((r_max_left - r_min_left) / h_r_left) + 1, dtype=np.float32)
    r_first_peak_left = np.linspace(r_min_first_peak_left, r_max_first_peak_left, int((r_max_first_peak_left -
                                                                                       r_max_left)
                                                                                      / h_r_the_first_peak_left),
                                    dtype=np.float32)
    r_top_of_the_first_peak = np.linspace(r_min_top_of_the_first_peak, r_max_top_of_the_first_peak,
                                          int((r_max_top_of_the_first_peak - r_max_first_peak_left)
                                              / h_r_top_of_the_first_peak), dtype=np.float32)
    r_first_peak_right = np.linspace(r_min_first_peak_right, r_max_first_peak_right,
                                     int((r_max_first_peak_right - r_max_top_of_the_first_peak)
                                         / h_r_the_first_peak_right), dtype=np.float32)
    r_right = np.linspace(r_min_right, r_max_right, int((r_max_right - r_max_first_peak_right) / h_r_right),
                          dtype=np.float32)
    r = np.r_[r_left, r_first_peak_left, r_top_of_the_first_peak, r_first_peak_right, r_right]
    # конкатенация массивов аргументов корреляционной функции
    #  ----------------------------- </параметры корреляционной функции> -----------------------------------------------
    #  ----------------------------- <отображение процента выполнения> -------------------------------------------------
    #  -------------------------------- <отображение прогресса разыгрывания состояний> ---------------------------------
    play_out_execution_progress = ExecutionProgress(output_play_out_progress_to_console, play_out_lower_bound_progress,
                                                    play_out_number_of_decimal_places)
    #  -------------------------------- </отображение прогресса разыгрывания состояний> --------------------------------
    #  -------------------------------- <отображение прогресса расчёта корреляционной функции> -------------------------
    pair_corr_func_execution_progress = ExecutionProgress(output_pair_corr_func_progress_to_console,
                                                          pair_corr_func_lower_bound_progress,
                                                          pair_corr_func_number_of_decimal_places)
    #  -------------------------------- </отображение прогресса расчёта корреляционной функции> ------------------------
    #  ----------------------------- </отображение процента выполнения> ------------------------------------------------
    # --------------------------- </вычисляемые константы> -------------------------------------------------------------
    # --------------------------- <вычисляем корреляционнную функцию> --------------------------------------------------
    pair_corr_func = np.zeros((len(T), len(r)))  # инициализируем массив с корреляционной функцией
    for i in range(len(T)):
        print("Начат расчёт при T = " + str(T[i] * T0) + " K")
        start_play_out_time = time.time()
        print("Дата и время начала расчёта: " + time.strftime("%D %H:%M:%S", time.localtime(start_play_out_time))
              + " hh:mm:ss")
        molecules_ensemble = np.zeros((M - M_relax, N, 3), dtype=np.float32)
        # содержит положения всех частиц во всех состояниях
        molecules_ensemble[0] = populate_cube(N_coordinate, L / N_coordinate)
        # заполняем куб частицами (начальное расположение)
        #  -------------------------- <разыгрываем состояния> ----------------------------------------------------------
        print("Разыгрываем состояния...")
        molecules_ensemble = play_out_states(molecules_ensemble, M, N, delta, L, T[i], play_out_execution_progress,
                                             play_out_description)
        end_play_out_time = time.time()
        print("Состояния разыграны")
        print(
            "Время разыгрывания состояний: " + str(datetime.timedelta(seconds=end_play_out_time - start_play_out_time))
            + " hh:mm:ss")
        print("Дата и время конца разыгрывания состояний: " +
              time.strftime("%D %H:%M:%S", time.localtime(end_play_out_time)) + " hh:mm:ss")
        #  -------------------------- </разыгрываем состояния> ---------------------------------------------------------
        start_corr_func_calc_time = time.time()
        print("Ищем парную корреляционную функцию...")
        if device_to_calculate_pair_correlation_function == CalculateOn.CPU:
            pair_corr_func[i] = calculate_pair_correlation_function_on_cpu(molecules_ensemble, r, delta_r, L, n,
                                                                           pair_corr_func_execution_progress,
                                                                           pair_corr_func_description)
        elif device_to_calculate_pair_correlation_function == CalculateOn.GPU:
            pair_corr_func[i] = calculate_pair_correlation_function_on_gpu(molecules_ensemble, r, delta_r, L, n,
                                                                           pair_corr_func_execution_progress, block_dim,
                                                                           pair_corr_func_description)
        end_corr_func_calc_time = time.time()
        #  -------------------------- <сохраняем массив с корреляционной функцией> -------------------------------------
        folder_name = "corr_func_arrays/M=" + str(M) + "/N=" + str(N)
        if not path.exists(folder_name):
            makedirs(folder_name)
        np.save(folder_name + "/corr_func M=" + str(M) + ", N=" + str(N) + ", T=" + str(T[i]), pair_corr_func[i])
        #  -------------------------- </сохраняем массив с корреляционной функцией> ------------------------------------
        print("Парная корреляционная функция найдена")
        print("Время расчёта корреляционной функции: "
              + str(datetime.timedelta(seconds=end_corr_func_calc_time - start_corr_func_calc_time)) + " hh:mm:ss")
        print("Дата и время конца расчёта корреляционной функции: " +
              time.strftime("%D %H:%M:%S", time.localtime(end_corr_func_calc_time)) + " hh:mm:ss")
        print("Общее время расчёта: " +
              str(datetime.timedelta(seconds=
                                     end_play_out_time - start_play_out_time + end_corr_func_calc_time
                                     - start_corr_func_calc_time)) + " hh:mm:ss")
        print("Отношение времени расчёта парной корреляционной функции ко времени разыгрывания состояний: "
              + str((end_corr_func_calc_time - start_corr_func_calc_time) / (end_play_out_time - start_play_out_time)))
        print("Расчёт при T = " + str(T[i] * T0) + " K завершён")
    # --------------------------- </вычисляем корреляционнную функцию> -------------------------------------------------
    # --------------------------- <строим корреляционнную функцию> -----------------------------------------------------
    for i in range(len(T)):
        plot(r, pair_corr_func[i], "g(r*), парная корреляционная функция " + chosen_noble_gas.value.name_ru +
             "а при T = " + str(T[i] * T0) + " K", "r*", "g(r*)")
    # --------------------------- </строим корреляционнную функцию> ----------------------------------------------------
