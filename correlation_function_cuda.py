import numpy as np
import pycuda.driver as drv
from pycuda.autoinit import context
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code_template = """
  #include <stdio.h>

__global__ void kernel(float* molecules, float* parameters_array, int* N) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int mdim = %(MDIM)s;
  const int ndim = %(NDIM)s;
  if (idy < mdim && idx < ndim) {
  float L = parameters_array[2];
  float focused_molecules[%(NDIM)s];
  float x_shift = L / 2 - molecules[idx + idy * ndim];
  float y_shift = L / 2 - molecules[idx + 1 + idy * ndim];
  float z_shift = L / 2 - molecules[idx + 2 + idy * ndim];
  float x, y, z;
  for (int i = 0; i < ndim; i += 3) {
        x = molecules[i + idy * ndim];
        y = molecules[i + 1 + idy * ndim];
        z = molecules[i + 2 + idy * ndim];
        if (x + x_shift >= L) {
            focused_molecules[i] = x + x_shift - L;
        } else if (x + x_shift < 0) {
            focused_molecules[i] = x + x_shift + L;
        } else {
            focused_molecules[i] = x + x_shift;
        }
        if (y + y_shift >= L) {
            focused_molecules[i + 1] = y + y_shift - L;
        } else if (y + y_shift < 0) {
            focused_molecules[i + 1] = y + y_shift + L;
        } else {
            focused_molecules[i + 1] = y + y_shift;
        }
        if (z + z_shift >= L) {
            focused_molecules[i + 2] = z + z_shift - L;
        } else if (z + z_shift < 0) {
            focused_molecules[i + 2] = z + z_shift + L;
        } else {
            focused_molecules[i + 2] = z + z_shift;
        }
  }
  float d = 0;
  for (int i = 0; i < ndim; i += 3) {
      if (i != idx) {
            d = sqrt(pow(focused_molecules[i] - L / 2, 2) + pow(focused_molecules[i + 1] 
            - L / 2, 2) + pow(focused_molecules[i + 2] - L / 2, 2));
            if ((parameters_array[0] - parameters_array[1]) < d && d < (parameters_array[0] + parameters_array[1])) {
                atomicAdd(N, 1);
            } 
      }
  }
  //free(focused_molecules);
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
    :param execution_percentage: класс, хранящий параметры вывода процента выполнения в консол
    :return: массив значений корреляционной функции
    """
    pair_corr_func = np.zeros(len(r))
    # --------------------------- <настраиваем ядро GPU> ---------------------------------------------------------------
    block_dim = (32, 32, 1)  # размерность блока
    dx, mx = divmod(len(molecules_ensemble[0]), block_dim[0])
    dy, my = divmod(len(molecules_ensemble), block_dim[1])
    grid_dim = ((dx + int(mx > 0)) * block_dim[0], (dy + int(my > 0)) * block_dim[1])  # размерность сетки
    kernel_code = kernel_code_template % {
        "MDIM": len(molecules_ensemble),
        "NDIM": 3 * len(molecules_ensemble[0]),
    }
    mod = SourceModule(kernel_code)
    calculate = mod.get_function("kernel")
    # --------------------------- </настраиваем ядро GPU> --------------------------------------------------------------
    parameters_array = np.array([0, delta_r / 2, L], dtype=np.float32)
    delta_N = drv.managed_zeros(shape=1, dtype=np.int32, mem_flags=drv.mem_attach_flags.GLOBAL)
    # инициализируем delta_N в общей памяти
    for i in range(len(r)):
        parameters_array[0] = r[i]
        calculate(drv.In(molecules_ensemble.flatten()), drv.In(parameters_array), delta_N,
                  block=block_dim, grid=grid_dim)
        context.synchronize()
        delta_N[0] = delta_N[0] / (len(molecules_ensemble) * len(molecules_ensemble[0]))
        # среднее количество частиц в шаровом слое
        pair_corr_func[i] = delta_N[0] / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
        delta_N[0] = 0
    return pair_corr_func
