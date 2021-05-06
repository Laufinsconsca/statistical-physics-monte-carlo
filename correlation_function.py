import numba
from numba import prange
import numpy as np
import pycuda.driver as drv
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
# noinspection PyUnresolvedReferences
import pycuda.autoinit
from functions import distance, output_execution_progress
from functions import focus_on_given_molecule


@numba.njit(cache=True, parallel=True)
def calculate_pair_correlation_function_on_cpu(molecules_ensemble, r, delta_r, L, n, execution_progress_struct):
    """
    Вычисление парной корреляционной функции

    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_progress_struct: класс типа ExecutionProgress, хранящий параметры отображения процента выполнения
    :return: массив значений корреляционной функции
    """
    pair_corr_func = np.zeros(len(r))
    if execution_progress_struct.output_progress_to_console:
        progress = 0
        h_p = 100 / len(r)
        p = 1  # период отображения процента выполнения (в итерациях)
        while execution_progress_struct.lower_bound > h_p * p:
            p += 1
        for i in range(len(r)):
            delta_N = 0
            for j in prange(len(molecules_ensemble)):
                for k in range(len(molecules_ensemble[0])):
                    delta_N += number_of_molecules_in_a_spherical_layer(molecules_ensemble[j], r[i], delta_r, k, L)
            progress += h_p
            if i % p == 0:
                output_execution_progress(execution_progress_struct, "Вычисление корреляционной функции",
                                          progress)
            delta_N = delta_N / (len(molecules_ensemble) * len(molecules_ensemble[0]))
            # среднее количество частиц в шаровом слое
            pair_corr_func[i] = delta_N / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
    else:
        for i in prange(len(r)):
            delta_N = 0
            for j in range(len(molecules_ensemble)):
                for k in range(len(molecules_ensemble[0])):
                    delta_N += number_of_molecules_in_a_spherical_layer(molecules_ensemble[j], r[i], delta_r, k, L)
            delta_N = delta_N / (len(molecules_ensemble) * len(molecules_ensemble[0]))
            # среднее количество частиц в шаровом слое
            pair_corr_func[i] = delta_N / (4 * np.pi * r[i] * r[i] * delta_r * n)  # корреляционная функция
    return pair_corr_func


@numba.njit(cache=True)
def number_of_molecules_in_a_spherical_layer(molecules, r, delta_r, num, L):
    """
    Вычисление количества молекул в сферическом слое выделенной молекулы

    :param molecules: массив молекул в некотором состоянии
    :param r: расстояние, на которое отстоит середина толщины сферического слоя от выделенной молекулы
    :param delta_r: толщина сферического слоя
    :param num: номер выделенной молекулы
    :param L: длина ребра ячейки моделирования
    :return: количество молекул в сферическом слое выделенной молекулы
    """
    focused_molecules = focus_on_given_molecule(molecules, num, L)
    # фокусируемся на выделенной молекуле (мысленно помещаем её в центр ячейки моделирования)
    # это необходимо для учёта частиц из соседних ячеек
    N = 0
    for i in range(len(molecules)):
        if i != num and (r - delta_r / 2) < distance(focused_molecules[num], focused_molecules[i]) < (r + delta_r / 2):
            N += 1
    return N


kernel_code = """
__global__ void kernel(float* molecules, int* N, double* r, double delta_r, double L, int rdim, int mdim, int ndim, 
unsigned long long int* progress, int is_output_progress_to_console, float play_out_lower_bound_progress, 
char* progress_string) {
  const int idx = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
  const int idy = threadIdx.y + blockDim.y * blockIdx.y;
  const int idz = threadIdx.z + blockDim.z * blockIdx.z;
  if (idx < ndim && idy < rdim && idz < mdim) {
  double progress_max = rdim*mdim*ndim;
  double x_shift = L / 2 - molecules[idx + idz * ndim];
  double y_shift = L / 2 - molecules[idx + 1 + idz * ndim];
  double z_shift = L / 2 - molecules[idx + 2 + idz * ndim];
  double x, y, z, d, temp;
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
    if ((progress[1] - progress[0]) <= play_out_lower_bound_progress) {
        atomicExch(&progress[2], 0);
    } else {
        atomicExch(&progress[2], !progress[2]);
        atomicExch(&progress[0], progress[1]);
    }
    if (progress[2] && idy % blockDim.y == 0){
        printf(progress_string, 300*progress[1]/progress_max);
    }
  }
  }
}
"""


def calculate_pair_correlation_function_on_gpu(molecules_ensemble, r, delta_r, L, n, execution_progress_struct,
                                               block_dim):
    """
    Вычисление парной корреляционной функции

    :param block_dim:
    :param molecules_ensemble: набор частиц во всех оставшихся состояних, по которым усредняем
    :param r: массив аргументов функции
    :param delta_r: толщина шарового слоя
    :param L: длина ребра ячейки моделирования
    :param n: средняя концентрация
    :param execution_progress_struct: класс, хранящий параметры вывода процента выполнения в консоль
    :return: массив значений корреляционной функции
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
    progress = drv.managed_zeros(shape=4, dtype=np.uint64, mem_flags=drv.mem_attach_flags.GLOBAL)
    progress[2] = 0
    progress_string = bytearray("Вычисление корреляционной функции: %."
                                + str(execution_progress_struct.number_of_decimal_places) + "f%%\n", 'utf-8')
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
              progress,
              np.int32(int(execution_progress_struct.output_progress_to_console)),
              np.float32(np.float32(len(molecules_ensemble) * len(molecules_ensemble[0]) * len(r)
                                    * execution_progress_struct.lower_bound) / 100),
              d_progress_string,
              block=block_dim, grid=grid_dim)
    context.synchronize()
    pair_corr_func = delta_N / (4 * np.pi * r * r * delta_r * n * len(molecules_ensemble) * len(molecules_ensemble[0]))
    # корреляционная функция
    return pair_corr_func


