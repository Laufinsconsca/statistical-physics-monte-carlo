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
    PAIR CORRELATION FUNCTION SIMULATION FOR LIQUID NOBLE GASES
    Brief:
    The pair correlation function shows the ratio of the concentration at a given distance to the average concentration.
    Near zero, the function is zero, since the particles cannot come close to each other.
    When the distance between particles tends to +∞, the concentration tends to the average, so the ratio tends to 1.
    The Lennard-Jones potential is used to describe the interaction between particles.
    """
    #  --------------------------- <common constants> ------------------------------------------------------------------
    k = 1.38e-23  # Boltzmann constant, J/K
    device_to_calculate_pair_correlation_function = CalculateOn.CPU
    # select the CPU or a GPU for calculating the pair correlation function (you can change the GPU device by setting an
    # integer value up to the environment variable CUDA_DEVICE, the first CUDA-supported device is selected by default)
    # (be aware: GPU calculation is supported only by Nvidia graphics cards)
    # to calculate on a GPU device you also need to add the path of cl.exe to the Path environment variable,
    # example of the cl.exe location below:
    # C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64
    block_dim = (2, 2, 64)  # the block dimension (you need to set in the case of GPU computing)
    # the maximum block dimension is limited by the GPU capabilities (both each of the dimensions and their product)
    #  -------------------------- </common constants> ------------------------------------------------------------------
    #  -------------------------- <simulation constants> ---------------------------------------------------------------
    N = 5 ** 3  # the number of particles in one cell (must be the cube of a natural number)
    M = 50000  # the number of states to play out
    h_T = 0.1  # the unitless temperature step
    delta = 0.1  # the molecule random shift constant
    M_relax = 49900  # the number of states to be eliminated (default 1000)
    #  ----------------------------- <gas parameters> ------------------------------------------------------------------
    temperature = 85  # the gas temperature, K
    chosen_noble_gas = ChooseNobleGas.ARGON  # argon was chosen
    #  ----------------------------- </gas parameters> -----------------------------------------------------------------
    #  ----------------------------- <pair correlation function parameters> --------------------------------------------
    h_r_left = 0.1  # the pair correlation function argument change step until the first peak is reached
    h_r_the_first_peak_left = 0.01  # the pair correlation function argument change step along the first peak
    # to the left of the top
    h_r_top_of_the_first_peak = 0.001  # the pair correlation function argument change step at the top of the first peak
    h_r_the_first_peak_right = 0.01  # the pair correlation function argument change step along the first peak
    # to the right of the top
    h_r_right = 0.01  # the pair correlation function argument change step after the first peak
    r_max_left = 0.8  # the end of zero of the pair correlation function
    r_max_first_peak_left = 1.05  # the end of the left side of the pair correlation function's first peak
    r_max_top_of_the_first_peak = 1.15  # the end of the top of the correlation function's peak
    r_max_first_peak_right = 1.3  # the end of the right side of the pair correlation function's first peak
    delta_r = 0.3  # the spherical layer thickness (the value is selected experimentally)
    #  ----------------------------- </pair correlation function parameters> -------------------------------------------
    #  -------------------------- </simulation constants> --------------------------------------------------------------
    #  -------------------------- <input arguments validation> ---------------------------------------------------------
    N_coordinate = check_n_is_integer(N ** (1. / 3))  # the particles number along any one coordinate
    # (with check the particles number to be an integer number)
    check_M_relax_less_than_M(M, M_relax)
    #  -------------------------- </input arguments validation> --------------------------------------------------------
    #  -------------------------- <displaying the execution progress> --------------------------------------------------
    #  ----------------------------- <displaying the playing states out progress> --------------------------------------
    output_play_out_progress_to_console = True  # allows you to track the program execution progress
    # (Be aware: reduces performance, at N <= 1000 by about 10%)
    play_out_lower_bound_progress = 5  # the number that sets the lower bound for the play out states
    # progress accuracy (in percent), if you reduce the parameter it reduces performance
    play_out_number_of_decimal_places = 2  # the max number of decimal places
    play_out_description = "Playing states out progress"
    #  ----------------------------- </displaying the playing states out progress> -------------------------------------
    #  ----------------------------- <displaying the calculating the pair correlation function progress> ---------------
    output_pair_corr_func_progress_to_console = True  # allows you to track the program execution progress
    # (Be aware: reduces performance, at N <= 1000 by about 10%)
    pair_corr_func_lower_bound_progress = 1  # the number that sets the lower bound for the pair correlation function
    # calculation progress accuracy (in percent), if you reduce the parameter it reduces performance
    pair_corr_func_number_of_decimal_places = 3  # the max number of decimal places
    pair_corr_func_description = "The pair correlation function calculation progress"
    #  ----------------------------- </displaying the calculating the pair correlation function progress> --------------
    #  -------------------------- <displaying the execution progress> --------------------------------------------------
    #  -------------------------- <constants to be calculated> ---------------------------------------------------------
    T0 = chosen_noble_gas.value.energy / k  # the characteristic temperature
    T_min = temperature / T0  # the initial calculation temperature
    T_max = temperature / T0  # the final calculation temperature
    T = np.linspace(T_min, T_max, int((T_max - T_min) / h_T + 1))  # the unitless temperature
    n_concentration = chosen_noble_gas.value.ro / chosen_noble_gas.value.mass  # the concentration
    n = n_concentration * (chosen_noble_gas.value.sigma ** 3)  # the unitless concentration
    # (sigma acts as the characteristic length)
    V = N / n  # the unitless volume
    L = V ** (1. / 3)  # the simulation cell edge length
    #  ----------------------------- <pair correlation function parameters> --------------------------------------------
    r_min_left = delta_r + 0.01  # the initial pair correlation function argument
    r_min_first_peak_left = r_max_left + h_r_the_first_peak_left  # the beginning of the left side
    # of the pair correlation function's first peak
    r_min_top_of_the_first_peak = r_max_first_peak_left + h_r_top_of_the_first_peak  # the beginning of the top
    # of the correlation function's peak
    r_min_first_peak_right = r_max_top_of_the_first_peak + h_r_the_first_peak_right  # the beginning of the right side
    # of the pair correlation function's first peak
    r_min_right = r_max_first_peak_right + h_r_right  # the beginning of the pair correlation function relaxation
    r_max_right = (L - delta_r) / 2  # the pair correlation function final argument
    # (the distance from the cube's center to the cube's face)
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
    # pair correlation function arguments arrays concatenation
    #  ----------------------------- </pair correlation function parameters> -------------------------------------------
    #  ----------------------------- <displaying the execution progress> -----------------------------------------------
    #  -------------------------------- <displaying the playing states out progress> -----------------------------------
    play_out_execution_progress = ExecutionProgress(output_play_out_progress_to_console, play_out_lower_bound_progress,
                                                    play_out_number_of_decimal_places)
    #  -------------------------------- </displaying the playing states out progress> ----------------------------------
    #  -------------------------------- <displaying the calculating the pair correlation function progress> ------------
    pair_corr_func_execution_progress = ExecutionProgress(output_pair_corr_func_progress_to_console,
                                                          pair_corr_func_lower_bound_progress,
                                                          pair_corr_func_number_of_decimal_places)
    #  -------------------------------- </displaying the calculating the pair correlation function progress> -----------
    #  ----------------------------- </displaying the execution progress> ----------------------------------------------
    # --------------------------- </constants to be calculated> --------------------------------------------------------
    # --------------------------- <pair correlation function calculation> ----------------------------------------------
    pair_corr_func = np.zeros((len(T), len(r)))  # an array to store the pair correlation function initialization
    for i in range(len(T)):
        print("The calculation was started, initial T is " + str(T[i] * T0) + " K")
        start_play_out_time = time.time()
        print("The date and the time when the calculation was started: " + time.strftime("%D %H:%M:%S",
                                                                                         time.localtime(
                                                                                             start_play_out_time))
              + " hh:mm:ss")
        molecules_ensemble = np.zeros((M - M_relax, N, 3), dtype=np.float32)
        # it contains the positions of all particles in all states
        molecules_ensemble[0] = populate_cube(N_coordinate, L / N_coordinate)
        # initial cube filling with particles
        #  -------------------------- <play out states> ----------------------------------------------------------------
        print("States are being played out...")
        molecules_ensemble = play_out_states(molecules_ensemble, M, N, delta, L, T[i], play_out_execution_progress,
                                             play_out_description)
        end_play_out_time = time.time()
        print("States have been played out")
        print(
            "The time spent on playing states out: "
            + str(datetime.timedelta(seconds=end_play_out_time - start_play_out_time)) + " hh:mm:ss")
        print("The date and the time when the states were played out: " +
              time.strftime("%D %H:%M:%S", time.localtime(end_play_out_time)) + " hh:mm:ss")
        #  -------------------------- </play out states> ---------------------------------------------------------------
        start_corr_func_calc_time = time.time()
        print("The pair correlation function are being calculated...")
        if device_to_calculate_pair_correlation_function == CalculateOn.CPU:
            pair_corr_func[i] = calculate_pair_correlation_function_on_cpu(molecules_ensemble, r, delta_r, L, n,
                                                                           pair_corr_func_execution_progress,
                                                                           pair_corr_func_description)
        elif device_to_calculate_pair_correlation_function == CalculateOn.GPU:
            pair_corr_func[i] = calculate_pair_correlation_function_on_gpu(molecules_ensemble, r, delta_r, L, n,
                                                                           pair_corr_func_execution_progress, block_dim,
                                                                           pair_corr_func_description)
        end_corr_func_calc_time = time.time()
        #  -------------------------- <storing the array with the pair correlation function> ---------------------------
        folder_name = "corr_func_arrays/M=" + str(M) + "/N=" + str(N)
        if not path.exists(folder_name):
            makedirs(folder_name)
        np.save(folder_name + "/corr_func M=" + str(M) + ", N=" + str(N) + ", T=" + str(T[i]), pair_corr_func[i])
        #  -------------------------- </storing the array with the pair correlation function> --------------------------
        print("The pair correlation function have been calculated")
        print("The time spent on pair correlation function calculation: "
              + str(datetime.timedelta(seconds=end_corr_func_calc_time - start_corr_func_calc_time)) + " hh:mm:ss")
        print("The date and the time when the pair correlation function was calculated: " +
              time.strftime("%D %H:%M:%S", time.localtime(end_corr_func_calc_time)) + " hh:mm:ss")
        print("The total calculation time: " +
              str(datetime.timedelta(seconds=
                                     end_play_out_time - start_play_out_time + end_corr_func_calc_time
                                     - start_corr_func_calc_time)) + " hh:mm:ss")
        print("The ratio of the time of pair correlation function calculation to the time of playing states out: "
              + str((end_corr_func_calc_time - start_corr_func_calc_time) / (end_play_out_time - start_play_out_time)))
        print("Calculation at T = " + str(T[i] * T0) + " K was completed")
    # --------------------------- </pair correlation function calculation> ---------------------------------------------
    # --------------------------- <the pair correlation function plotting> ---------------------------------------------
    for i in range(len(T)):
        plot(r, pair_corr_func[i], "g(r*), the pair correlation function of " + chosen_noble_gas.value.name_en
             + " at T = " + str(T[i] * T0) + " K", "r*", "g(r*)")
    # --------------------------- </the pair correlation function plotting> --------------------------------------------
