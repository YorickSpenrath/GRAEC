import abc
import math
from matplotlib import pyplot as plt
import numpy as np


class Timer:
    def __init__(self, t_e):
        self.t_e = t_e

    @abc.abstractmethod
    def get_fraction(self, time):
        pass


class LinearTimer(Timer):
    def __init__(self, end_fraction=1, **kwargs):
        super().__init__(**kwargs)
        self.end_fraction = end_fraction

    def get_fraction(self, time):
        return self.end_fraction * time / self.t_e


class ConstantTimer(Timer):
    def __init__(self, fraction, **kwargs):
        super().__init__(**kwargs)
        self.fraction = fraction

    def get_fraction(self, time):
        return self.fraction


class BaseConceptDrifter:

    def __init__(self, dimension_size, number_activities, number_of_months, timer):
        self.L = dimension_size
        self.n = number_activities
        self.t_e = number_of_months
        assert (isinstance(timer, Timer))
        self.timer = timer
        pass

    def tf(self, time):
        return self.timer.get_fraction(time)

    @abc.abstractmethod
    def is_short(self, time, publications, pages):
        pass

    def plot(self, resolution=1, inter_period=1, figure=None, style='horizontal'):
        assert (isinstance(inter_period, int))
        assert (inter_period >= 1)

        if figure is None:
            fig = plt.figure(figure)
        else:
            fig = figure
        number_of_plots = int(self.timer.t_e / inter_period) + 1

        if style == 'square':
            rows = int(number_of_plots ** 0.5)
            columns = math.ceil(number_of_plots / rows)
        elif style == 'horizontal':
            rows = 1
            columns = number_of_plots
        else:
            raise Exception('Not a known style')

        for i in range(number_of_plots):
            ax = fig.add_subplot(rows, columns, i + 1)
            assert (isinstance(ax, plt.Axes))

            xs = []
            ys = []
            t = i * inter_period
            for x in np.arange(start=0, stop=100, step=resolution):
                for y in np.arange(start=0, stop=100, step=resolution):
                    if self.is_short(t, x, y):
                        xs.append(x)
                        ys.append(y)
            ax.plot(xs, ys, 'k.')
            ax.set_title('t={}'.format(t), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, self.L)
            ax.set_ylim(0, self.L)
            if i == 0:
                ax.set_ylabel('Pages', fontdict={'fontsize': 8})
            if i == (number_of_plots // 2):
                ax.set_xlabel('Publications', fontdict={'fontsize': 8})
            ax.set_aspect('equal')
        if figure is None:
            plt.show()


class RadialConceptDrifter(BaseConceptDrifter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A = ((1 - 4 / (math.pi * (self.n + 1))) ** 0.5) * self.L
        self.B = 4 / (math.pi * (self.n + 1)) * (self.L ** 2)

    def inner(self, time):
        return self.A * self.tf(time)

    def outer(self, time):
        return (self.B + self.inner(time) ** 2) ** 0.5

    def is_short(self, time, publications, pages):
        return self.inner(time) <= (publications ** 2 + pages ** 2) ** 0.5 <= self.outer(time)


class LinearSumConceptDrifter(BaseConceptDrifter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.A = math.sqrt((self.n - 1) / (self.n + 1)) * self.L

    def is_short(self, time, publications, pages):
        return self.inner(time) <= publications + pages <= self.outer(time)

    def inner(self, time):
        return self.A * self.tf(time)

    def outer(self, time):
        return (2 / (self.n + 1) * self.L ** 2 + self.inner(time) ** 2) ** 0.5


class LinearConceptDrifter(BaseConceptDrifter):
    def is_short(self, time, publications, pages):
        return 0 <= (pages if self.use_pages else publications) - self.A * self.tf(time) <= self.L / (self.n + 1)

    def __init__(self, use_pages=True, **kwargs):
        super().__init__(**kwargs)
        self.use_pages = use_pages
        self.A = self.n / (self.n + 1) * self.L
