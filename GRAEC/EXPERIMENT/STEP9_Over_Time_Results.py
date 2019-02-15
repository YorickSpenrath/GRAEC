"""
Plots the F1 and accuracy scores over time
"""
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GRAEC.Functions import DataFrameOperations, Filefunctions

from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters

colours = ['r', 'orange', 'b'] * 3
markers = ['o'] * 3 + ['v'] * 3 + ['^'] * 3

metric_scores = ('accuracy', 'f1')


class PlotEvaluation:

    def __init__(self,
                 default_s,
                 parameters=('Beta', 'Tau', 'S'),
                 window=1 if Parameters.Demo else 100,
                 offset=0,
                 step=1 if Parameters.Demo else 14,
                 metrics=metric_scores,
                 plot_naive=True,
                 plot_previous=True,
                 ):

        assert (isinstance(default_s, int))
        self.default_S = default_s
        self.parameters = parameters
        assert (isinstance(window, int))
        self.window = window
        assert (isinstance(offset, int))
        self.offset = offset
        assert (isinstance(step, int))
        self.step = step
        self.metrics = metrics
        assert (isinstance(plot_naive, bool))
        self.plot_naive = plot_naive
        assert (isinstance(plot_previous, bool))
        self.plot_previous = plot_previous

    def run(self):

        for metric in self.metrics:

            naive_signal = self.get_naive_signal(self.default_S, metric)
            previous_signal = self.get_previous_signal(self.default_S, metric)

            for param_i, param in enumerate(self.parameters):
                signals = self.get_signals(parameter=param, metric=metric)
                fig, ax = plt.subplots()

                for i, v in enumerate(signals):
                    ax.plot(signals[v]['x'],
                            signals[v]['y'],
                            markers[i] + ('' if Parameters.Demo else '-'),
                            color=colours[i],
                            label=str(v))
                    ax.set_title(Parameters.Tex_dict[param])

                if self.plot_naive:
                    ax.plot(naive_signal['x'],
                            naive_signal['y'],
                            linestyle=':',
                            color='k',
                            label='N{}'.format(self.default_S))
                if self.plot_previous:
                    ax.plot(previous_signal['x'],
                            previous_signal['y'],
                            linestyle='',
                            marker='x',
                            color='k',
                            label='R{}'.format(self.default_S))
                ax.legend(ncol=math.ceil(len(signals) / 3))
                if Parameters.Demo:
                    ax.set_ylim(0, 1.0)
                    ax.set_xlabel('Month')
                    ax.set_ylabel('{} per month'.format(metric, self.window))
                else:
                    ax.set_ylim(0, 0.5)
                    ax.set_xlabel('Day')
                    ax.set_ylabel('{} over past {} days'.format(metric, self.window))
                fig.set_size_inches(10 / 2.56, 10 / 2.56)
                fn = Name_functions.parameter_evaluation_figure(parameter=param,
                                                                metric=metric)
                plt.savefig(fn, bbox_inches='tight')
                plt.close()

    def get_previous_signal(self, s, metric):
        return self.get_naive_or_previous_signal(s, metric, 'Previous')

    def get_naive_signal(self, s, metric):
        return self.get_naive_or_previous_signal(s, metric, 'Naive')

    def get_naive_or_previous_signal(self, s, metric, naive_or_previous):
        assert (s in Parameters.S_values)
        assert (naive_or_previous == 'Naive' or naive_or_previous == 'Previous')
        fn = Name_functions.parameter_evaluation_evaluation_metric_file(naive_or_previous)
        df = DataFrameOperations.import_df(fn, dtype=Parameters.parameter_evaluation_dtypes)

        # Select relevant values
        df = df[df['S'] == s]

        # Input for transform_to_signal function
        raw_signal = df[['NumEntries', metric, 'Day']]
        x, y = self.transform_to_signal(df=raw_signal)
        return {'x': x, 'y': y}

    def transform_to_signal(self, df):
        assert (isinstance(df, pd.DataFrame))
        assert ('Day' in df.columns)
        assert ('NumEntries' in df.columns)
        assert (len(df.columns == 3))
        all_columns = df.columns.tolist()
        all_columns.remove('Day')
        all_columns.remove('NumEntries')
        metric = all_columns[0]
        raw_signal = df.copy()
        assert (isinstance(raw_signal, pd.DataFrame))
        raw_signal['Weighted_{}'.format(metric)] = raw_signal[metric] * raw_signal['NumEntries']

        x = []
        y = []
        w_end = self.offset + self.window
        end = np.max(raw_signal['Day'])
        while w_end < end:
            data = raw_signal[(raw_signal['Day'] >= w_end - self.window) & (raw_signal['Day'] < w_end)]
            if sum(data['NumEntries']) != 0:
                x.append(w_end)
                y.append(sum(data['Weighted_{}'.format(metric)]) / sum(data['NumEntries']))
            w_end += self.step

        return x, y

    def get_signals(self, parameter, metric, values=None):
        assert (isinstance(parameter, str))

        fn = Name_functions.parameter_evaluation_evaluation_metric_file(parameter)
        df = DataFrameOperations.import_df(fn, dtype=Parameters.parameter_evaluation_dtypes)
        df.dropna(axis=0, how='any', subset=['NumEntries', metric], inplace=True)
        signals = dict()

        if values is None:
            values = df[parameter].unique()

        for v in values:
            raw_signal = df[df[parameter] == v]
            raw_signal = raw_signal[['Day', 'NumEntries', metric]]
            x, y = self.transform_to_signal(df=raw_signal)
            signals[v] = dict()
            signals[v]['x'] = x
            signals[v]['y'] = y
            signals[v]['parameter'] = parameter
            signals[v]['value'] = v
        return signals


def undo():

    for metric in metric_scores:
        Filefunctions.delete(Name_functions.parameter_metric_evaluation_folder(metric))


def run_on():
    run()


def restart():
    undo()
    run_on()


def run():
    with open(Name_functions.best_graec(), 'r') as rf:
        (S, B, T, P) = rf.readline()[:-1].split(';')
    PlotEvaluation(int(S)).run()


if __name__ == '__main__':
    run()
