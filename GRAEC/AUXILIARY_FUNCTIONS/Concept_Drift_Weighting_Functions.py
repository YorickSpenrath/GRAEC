"""
Weighting functions that are used in the GRAEC
"""

import matplotlib.pyplot as plt
import numpy as np


class BEPTScoring:
    def __init__(self, exponential, width, period, factor):
        self.beta = exponential
        self.epsilon = width
        self.p = period
        self.tau = factor

    # def get_avg_weight(self, x):
    #     assert(isinstance(x, float))
    #     self.weight(x)

    def weight(self, x):
        assert (isinstance(x, int) or isinstance(x, float))
        x = float(x)
        if self.tau == 0:
            return self.exponential(x)
        else:
            return self.exponential(x) + self.tau * self.seasonality(x)

    def exponential(self, x):
        try:
            return np.exp(- self.beta * x)
        except OverflowError:
            print(self.beta)
            print(x)
            raise OverflowError

    def seasonality(self, x):
        return np.exp(- np.power(self.__augmented_modulo(x), 2) / self.epsilon)

    def __augmented_modulo(self, x):
        return (x % self.p) if (x % self.p < self.p / 2) else (self.p - x % self.p)

    def to_string(self):
        return 'exp(-{:.2f} x) + {:.2f} * exp(- eta_{:.2f}(x)^2 / {:.2f})'.format(self.beta, self.tau, self.p,
                                                                                  self.epsilon)

    def plot(self):
        x = range(1000)
        y_season = [self.seasonality(i) for i in x]
        y_exp = [self.exponential(i) for i in x]
        y_total = [self.weight(i) for i in x]
        fig = plt.figure()
        fig.suptitle('Visual representation of weighting function')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Weight')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

        for (i, y) in enumerate([y_exp, y_season, y_total]):
            axs = fig.add_subplot(3, 1, i + 1)
            plt.plot(x, y)
            axs.text(900, 0.4 if i < 2 else 0.8, r"$\Omega{}$".format("_{}".format(i) if i < 2 else ''),
                     fontdict={'fontsize': 20})
            axs.set_xlim(0, 1000)
            axs.set_ylim(0, 1 if i < 2 else 2)
            axs.set_yticks([0, 1] if i < 2 else [0, 1, 2])
            if i != 2:
                axs.set_xticks([])

        plt.show()


class PeriodScoring:
    def __init__(self, beta, p, tau, s, eps_base=-6):
        assert (beta >= 0)
        self.beta = beta

        assert (tau >= 0)
        self.tau = tau
        assert (tau == 0 or p >= 10 ** eps_base)
        self.p = p
        assert (s > 0)
        self.S = s
        self.eps_base = eps_base

    def get_weights(self, data_time):
        period = int(data_time / self.S)
        weights = dict()
        k = 1
        while k <= period and -(k - 1) * self.beta >= self.eps_base:
            weights[period - k] = 10 ** (-(k - 1) * self.beta)
            k += 1

        if self.tau == 0:
            return weights

        k = 1
        while int(data_time - k * self.p) > 0:
            ssp = int((data_time - k * self.p) / self.S)
            weights[ssp] = weights.get(ssp, 0) + self.tau
            k += 1

        return weights

    def plot(self, in_ax=None):
        weights = self.get_weights(1000)
        if in_ax is None:
            _, ax = plt.subplots()
        else:
            ax = in_ax

        ax.plot(range(1000), [weights.get(y, 0) for y in range(1000)], '.')

        if in_ax is None:
            plt.show()

    def to_string(self):
        return '[S={}, Beta={}, Tau={}, P={}]'.format(self.S, self.beta, self.tau, self.p)
