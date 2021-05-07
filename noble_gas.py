from enum import Enum


class NobleGas(object):
    """
    Класс, описывающий параметры благородного газа
    """
    mass = 0
    energy = 0
    sigma = 0
    ro = 0
    name_ru = ""
    name_en = ""

    def __init__(self, mass, energy, sigma, ro, name_ru, name_en):
        self.mass = mass
        self.energy = energy
        self.sigma = sigma
        self.ro = ro
        self.name_ru = name_ru
        self.name_en = name_en


class ChooseNobleGas(Enum):
    NEON = NobleGas(20.1797 * 1.660539e-27, 3.1e-3 * 1.6e-19, 2.74e-10, 900, "неон", "neon")
    ARGON = NobleGas(39.948 * 1.660539e-27, 1.04e-2 * 1.6e-19, 3.4e-10, 1401, "аргон", "argon")
    KRYPTON = NobleGas(83.798 * 1.660539e-27, 1.4e-2 * 1.6e-19, 3.65e-10, 2412, "криптон", "krypton")
    XENON = NobleGas(131.293 * 1.660539e-27, 1.997e-2 * 1.6e-19, 3.98e-10, 2942, "ксенон", "xenon")
