import numba
import numpy as np
# noinspection PyUnresolvedReferences
import pycuda.autoinit
import pycuda.driver as drv
from numba import prange
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

from execution_progress import output_execution_progress


@numba.njit(cache=True, parallel=True)
def calculate_pair_correlation_function_on_cpu(molecules_ensemble, r, delta_r, L, n, execution_progress_struct,
                                               description):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояниях, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_progress_struct: класс типа ExecutionProgress, хранящий параметры отображения процента выполнения
    :return: массив значений корреляционной функции
    :param description: описание выполняемого процесса
    """
    d = 0
    pair_corr_func = np.zeros(len(r))
    progress = 0
    h_p = 100 / len(r)
    p = 1  # период отображения процента выполнения (в итерациях)
    while execution_progress_struct.lower_bound > h_p * p:
        p += 1
    for i in range(len(r)):
        delta_N = 0
        for j in prange(len(molecules_ensemble)):
            for k in range(len(molecules_ensemble[0])):
                x_shift = L / 2 - molecules_ensemble[j][k][0]
                y_shift = L / 2 - molecules_ensemble[j][k][1]
                z_shift = L / 2 - molecules_ensemble[j][k][2]
                for q in range(len(molecules_ensemble[0])):
                    if k != q:
                        x = molecules_ensemble[j][q][0]
                        y = molecules_ensemble[j][q][1]
                        z = molecules_ensemble[j][q][2]
                        if x + x_shift >= L:
                            temp = x + x_shift - L
                        elif x + x_shift < 0:
                            temp = x + x_shift + L
                        else:
                            temp = x + x_shift
                        d += (temp - L / 2) ** 2
                        if y + y_shift >= L:
                            temp = y + y_shift - L
                        elif y + y_shift < 0:
                            temp = y + y_shift + L
                        else:
                            temp = y + y_shift
                        d += (temp - L / 2) ** 2
                        if z + z_shift >= L:
                            temp = z + z_shift - L
                        elif z + z_shift < 0:
                            temp = z + z_shift + L
                        else:
                            temp = z + z_shift
                        d += (temp - L / 2) ** 2
                        if (r[i] - delta_r / 2) < np.sqrt(d) < (r[i] + delta_r / 2):
                            delta_N += 1
                        d = 0
        if execution_progress_struct.output_progress_to_console:
            progress += h_p
            if i % p == 0:
                output_execution_progress(execution_progress_struct, description,
                                          progress)
        pair_corr_func[i] = delta_N
    return pair_corr_func / (4 * np.pi * r * r * delta_r * n
                             * len(molecules_ensemble) * len(molecules_ensemble[0]))


kernel_code = """
__global__ void kernel(float* molecules, int* N, float* r, float delta_r, float L, int rdim, int mdim, int ndim, 
unsigned long long int* progress, int is_output_progress_to_console, float lower_bound_progress, 
char* progress_string) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;
  if (idx < ndim && idy < rdim && idz < mdim) {
  double progress_max = (((double)rdim)*mdim*ndim)/300;
  float x_shift = L / 2 - molecules[idx + idz * ndim];
  float y_shift = L / 2 - molecules[idx + 1 + idz * ndim];
  float z_shift = L / 2 - molecules[idx + 2 + idz * ndim];
  float x, y, z, d, temp;
  for (int i = 0; i < ndim; i += 3) {
        if (i != idx) {
        x = molecules[i + idz * ndim];
        y = molecules[i + 1 + idz * ndim];
        z = molecules[i + 2 + idz * ndim];
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
        if ((r[idy] - delta_r) < d && d < (r[idy] + delta_r)) {
            atomicAdd(&N[idy], 1);
        }
        d = 0;
        } 
  }
  if (is_output_progress_to_console) {
    atomicAdd(&progress[1], 1);
    if ((progress[1] - progress[0]) <= lower_bound_progress) {
        atomicExch(&progress[2], 0);
    } else {
        atomicExch(&progress[2], !progress[2]);
        atomicExch(&progress[0], progress[1]);
    }
    if (progress[2] && idy % blockDim.y == 0){
        printf(progress_string, progress[1]/progress_max);
    }
  }
  }
}
"""


def calculate_pair_correlation_function_on_gpu(molecules_ensemble, r, delta_r, L, n, execution_progress_struct,
                                               block_dim, description):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_progress_struct: класс, хранящий параметры вывода процента выполнения в консоль
    :return: массив значений корреляционной функции
    :param block_dim: размерность блока (максимальная размерность блока ограничена возможностями GPU)
    :param description: описание выполняемого процесса
    """
    # --------------------------- <настраиваем ядро GPU> ---------------------------------------------------------------
    dx, mx = divmod(len(molecules_ensemble[0]), block_dim[0])
    dy, my = divmod(len(r), block_dim[1])
    dz, mz = divmod(len(molecules_ensemble), block_dim[2])
    grid_dim = ((dx + int(mx > 0)) * block_dim[0], (dy + int(my > 0)) * block_dim[1], (dz + int(mz > 0)) * block_dim[2])
    # размерность сетки
    mod = SourceModule(kernel_code)
    calculate = mod.get_function("kernel")
    # --------------------------- </настраиваем ядро GPU> --------------------------------------------------------------
    delta_N = drv.managed_zeros(shape=len(r), dtype=np.int32, mem_flags=drv.mem_attach_flags.GLOBAL)
    progress = drv.managed_zeros(shape=3, dtype=np.uint64, mem_flags=drv.mem_attach_flags.GLOBAL)
    progress_string = bytearray(description + ": %."
                                + str(execution_progress_struct.number_of_decimal_places) + "f%%\n", 'utf-8')
    d_progress_string = drv.mem_alloc(len(progress_string))
    drv.memcpy_htod(d_progress_string, progress_string)
    calculate(drv.In(molecules_ensemble.flatten()),
              delta_N,
              drv.In(r),
              np.float32(delta_r / 2),
              np.float32(L),
              np.int32(len(r)),
              np.int32(len(molecules_ensemble)),
              np.int32(3 * len(molecules_ensemble[0])),
              progress,
              np.int32(int(execution_progress_struct.output_progress_to_console)),
              np.float32(np.float32(len(molecules_ensemble) * len(molecules_ensemble[0]) * len(r)
                                    * execution_progress_struct.lower_bound) / 100),
              d_progress_string,
              block=block_dim, grid=grid_dim)
    context.synchronize()
    return delta_N / (4 * np.pi * r * r * delta_r * n * len(molecules_ensemble) * len(molecules_ensemble[0]))
    # возвращаем корреляционную функцию
