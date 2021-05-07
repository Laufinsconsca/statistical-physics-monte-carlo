import matplotlib.pyplot as plt  # отрисовка графиков


def plot(x, f, title, x_label, y_label):
    """
    Построение графика

    :param x: массив аргументов функции
    :param f: массив значений функции
    :param title: заглавие графика
    :param x_label: обозначение шкалы абсцисс
    :param y_label: обозначение шкалы ординат
    """
    plt.figure(figsize=(11, 8), dpi=80)
    # plt.scatter(x, f, s=5) # построение графика в виде набора точек
    plt.plot(x, f)  # построение сплошного графика
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()
