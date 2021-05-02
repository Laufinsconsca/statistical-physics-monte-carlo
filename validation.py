import numpy as np


def check_n_is_integer(n):
    """
    Проверка числа на целочисленность с приведением к целому типу в случае возникновения ошибок расчёта < eps
    :param n: проверяемое число типа double
    :return: число типа int или исключение с выходом из программы
    """
    eps = 1e-14
    if abs(np.floor(n) - n) > eps and abs(np.ceil(n) - n) > eps:
        print("N должно быть кубом натурального числа")
        exit()
    if abs(np.floor(n) - n) < eps:
        return int(np.floor(n))
    elif abs(np.ceil(n) - n) < eps:
        return int(np.ceil(n))


def check_M_relax_less_than_M(M, M_relax):
    """
    Проверка, меньше ли число отсеиваемых состояний чем общее число разыгрываемых состояний

    :param M: общее число разыгрываемых состояний
    :param M_relax: число отсеиваемых состояний
    :return: выход из программы в случае невыполнения условия
    """
    if M_relax >= M:
        print("Число отсеиваемых состояний не должно превышать общее число разыгрываемых состояний")
        exit()
