import numpy as np


def check_n_is_integer(n):
    if abs(np.floor(n) - n) > 1e-14 and abs(np.ceil(n) - n) > 1e-14:
        print("N должно быть кубом натурального числа")
        exit()
    if abs(np.floor(n) - n) < 1e-14:
        return int(np.floor(n))
    elif abs(np.ceil(n) - n) < 1e-14:
        return int(np.ceil(n))
