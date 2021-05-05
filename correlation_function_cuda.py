import numpy as np
import pycuda.driver as drv
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
import pycuda.autoinit

kernel_code = """
  #include <stdio.h>
  #include "Windows.h"

__global__ void kernel(float* molecules, int* N, double r, double delta_r, double L, int mdim, int ndim, 
double* percentage_of_completion) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  if (idy < mdim && idx < ndim) {
  double x_shift = L / 2 - molecules[idx + idy * ndim];
  double y_shift = L / 2 - molecules[idx + 1 + idy * ndim];
  double z_shift = L / 2 - molecules[idx + 2 + idy * ndim];
  double x, y, z, d, temp;
  for (int i = 0; i < ndim; i += 3) {
        if (i != idx) {
        x = molecules[i + idy * ndim];
        y = molecules[i + 1 + idy * ndim];
        z = molecules[i + 2 + idy * ndim];
        if (x + x_shift >= L) {
            temp = x + x_shift - L;
        } else if (x + x_shift < 0) {
            temp = x + x_shift + L;
        } else {
            temp = x + x_shift;
        }
        d += pow(temp - L / 2, 2);
        if (y + y_shift >= L) {
            temp = y + y_shift - L;
        } else if (y + y_shift < 0) {
            temp = y + y_shift + L;
        } else {
            temp = y + y_shift;
        }
        d += pow(temp - L / 2, 2);
        if (z + z_shift >= L) {
            temp = z + z_shift - L;
        } else if (z + z_shift < 0) {
            temp = z + z_shift + L;
        } else {
            temp = z + z_shift;
        }
        d += pow(temp - L / 2, 2);
        d = sqrt(d);
        if ((r - delta_r) < d && d < (r + delta_r)) {
            atomicAdd(N, 1);
        }
        d = 0;
        } 
  }
  atomicAdd(percentage_of_completion, 1);
  }
}
"""


def calculate_pair_correlation_function(molecules_ensemble, r, delta_r, L, n, execution_percentage):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_percentage: класс, хранящий параметры вывода процента выполнения в консоль
    :return: массив значений корреляционной функции
    """
    pair_corr_func = np.zeros(len(r))
    # --------------------------- <настраиваем ядро GPU> ---------------------------------------------------------------
    block_dim = (32, 32, 1)  # размерность блока
    dx, mx = divmod(len(molecules_ensemble[0]), block_dim[0])
    dy, my = divmod(len(molecules_ensemble), block_dim[1])
    grid_dim = ((dx + int(mx > 0)) * block_dim[0], (dy + int(my > 0)) * block_dim[1])  # размерность сетки
    mod = SourceModule(kernel_code)
    calculate = mod.get_function("kernel")
    # --------------------------- </настраиваем ядро GPU> --------------------------------------------------------------
    delta_N = drv.managed_zeros(shape=1, dtype=np.int32, mem_flags=drv.mem_attach_flags.GLOBAL)
    percentage_of_completion = drv.managed_zeros(shape=1, dtype=np.float64, mem_flags=drv.mem_attach_flags.GLOBAL)
    h_p = 100 / len(r)
    p = 1  # период отображения процента выполнения (в итерациях)
    while execution_percentage.lower_bound > h_p * p:
        p += 1
    # инициализируем delta_N в общей памяти
    for i in range(len(r)):
        calculate(drv.In(molecules_ensemble.flatten()),
                  delta_N,
                  r[i],
                  np.float64(delta_r / 2),
                  np.float64(L),
                  np.int32(len(molecules_ensemble)),
                  np.int32(3 * len(molecules_ensemble[0])),
                  percentage_of_completion,
                  block=block_dim, grid=grid_dim)
        context.synchronize()
        delta_N_temp = delta_N[0] / np.float64((len(molecules_ensemble) * len(molecules_ensemble[0])))
        # среднее количество частиц в шаровом слое
        pair_corr_func[i] = delta_N_temp / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
        delta_N[0] = 0
        if execution_percentage.output_percentage_to_console and i % p == 0:
            percentage_of_completion_str = "Расчёт корреляционной функции: {:." \
                                           + str(execution_percentage.number_of_decimal_places) + "f}%"
            print(percentage_of_completion_str
                  .format(100 * percentage_of_completion[0] / (len(r) * len(molecules_ensemble)
                                                               * len(molecules_ensemble[0]))))
    return pair_corr_func
