import numpy as np
import pycuda.driver as drv
from pycuda.autoinit import context
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code_template = """
  #include <stdio.h>

__global__ void kernel(float* molecules, float* r, int* N) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int mdim = %(MDIM)s;
  const int ndim = %(NDIM)s;
  if (idy < mdim && idx < ndim) {
  float L = %(L)s;
  float x_shift = L / 2 - molecules[idx + idy * ndim];
  float y_shift = L / 2 - molecules[idx + 1 + idy * ndim];
  float z_shift = L / 2 - molecules[idx + 2 + idy * ndim];
  float x, y, z, d, temp;
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
        if ((r[0] - %(DELTA_R_2)s) < d && d < (r[0] + %(DELTA_R_2)s)) {
                atomicAdd(N, 1);
        }
        d = 0;
        } 
  }
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
        "L": L, "DELTA_R_2": delta_r/2
    }
    mod = SourceModule(kernel_code)
    calculate = mod.get_function("kernel")
    # --------------------------- </настраиваем ядро GPU> --------------------------------------------------------------
    r_ = np.zeros(1, dtype=np.float32)
    delta_N = drv.managed_zeros(shape=1, dtype=np.int32, mem_flags=drv.mem_attach_flags.GLOBAL)
    # инициализируем delta_N в общей памяти
    for i in range(len(r)):
        r_[0] = r[i]
        calculate(drv.In(molecules_ensemble.flatten()), drv.In(r_), delta_N, block=block_dim, grid=grid_dim)
        context.synchronize()
        delta_N[0] = delta_N[0] / (len(molecules_ensemble) * len(molecules_ensemble[0]))
        # среднее количество частиц в шаровом слое
        pair_corr_func[i] = delta_N[0] / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
        delta_N[0] = 0
    return pair_corr_func
