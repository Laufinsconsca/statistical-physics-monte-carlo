import numpy as np
import pycuda.driver as drv
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
import pycuda.autoinit

kernel_code = """
__global__ void kernel(float* molecules, int* N, double* r, double delta_r, double L, int rdim, int mdim, int ndim, 
double* progress, int is_output_percentage_to_console, float lower_bound, char* progress_string) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;
  if (idz < rdim && idy < mdim && idx < ndim) {
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
        if ((r[idz] - delta_r) < d && d < (r[idz] + delta_r)) {
            atomicAdd(&N[idz], 1);
            __threadfence_system();
        }
        d = 0;
        } 
  }
  atomicAdd(&progress[1], 1);
  if (is_output_percentage_to_console && idx % ndim == 0 && 300*(progress[1] - progress[0])/(rdim*mdim*ndim) > 
  lower_bound) {
        progress[0] = progress[1];
        printf(progress_string, 300*progress[1]/(rdim*mdim*ndim));
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
    :param execution_percentage: класс, хранящий параметры вывода процента выполнения в консоль
    :return: массив значений корреляционной функции
    """
    # --------------------------- <настраиваем ядро GPU> ---------------------------------------------------------------
    block_dim = (16, 16, 4)  # размерность блока
    dx, mx = divmod(len(molecules_ensemble[0]), block_dim[0])
    dy, my = divmod(len(molecules_ensemble), block_dim[1])
    dz, mz = divmod(len(r), block_dim[2])
    grid_dim = ((dx + int(mx > 0)) * block_dim[0], (dy + int(my > 0)) * block_dim[1], (dz + int(mz > 0)) * block_dim[2])
    # размерность сетки
    mod = SourceModule(kernel_code)
    calculate = mod.get_function("kernel")
    # --------------------------- </настраиваем ядро GPU> --------------------------------------------------------------
    delta_N = drv.managed_zeros(shape=len(r), dtype=np.int32, mem_flags=drv.mem_attach_flags.GLOBAL)
    percentage_of_completion = drv.managed_zeros(shape=2, dtype=np.float64, mem_flags=drv.mem_attach_flags.GLOBAL)
    progress_string = bytearray("Вычисление корреляционной функции: %."
                                + str(execution_percentage.number_of_decimal_places) + "f%%\n", 'utf-8')
    d_progress_string = drv.mem_alloc(len(progress_string))
    drv.memcpy_htod(d_progress_string, progress_string)
    calculate(drv.In(molecules_ensemble.flatten()),
              delta_N,
              drv.In(r),
              np.float64(delta_r / 2),
              np.float64(L),
              np.int32(len(r)),
              np.int32(len(molecules_ensemble)),
              np.int32(3 * len(molecules_ensemble[0])),
              percentage_of_completion,
              np.int32(int(execution_percentage.output_percentage_to_console)),
              np.float32(execution_percentage.lower_bound),
              d_progress_string,
              block=block_dim, grid=grid_dim)
    context.synchronize()
    pair_corr_func = delta_N / (4 * np.pi * r * r * delta_r * n * len(molecules_ensemble) * len(molecules_ensemble[0]))
    # корреляционная функция
    return pair_corr_func
