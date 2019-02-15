"""
Creates the data required for the plots over time
"""

import pandas as pd

from GRAEC.Functions import Filefunctions
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions, Metrics, Classifiers
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter as Di
from GRAEC.AUXILIARY_FUNCTIONS.Concept_Drift_Weighting_Functions import PeriodScoring
import numpy as np

from GRAEC.EXPERIMENT import STEP9_Over_Time_Results

all_labels = Parameters.all_labels


class CalculateDailyScores:
    """
    Creates a plot over time for given parameter variations
    # Experiment_name : Save location of the the experiment information
    Each parameter p in the keys of multi is tested for each value of multi[p], while keeping the
    other parameters constant, with values given by single[p]
    # Moment : name of original data
    """

    def __init__(self, single, multi, test_ids_fn):
        # Instance assertion
        assert (isinstance(single, dict))
        assert (isinstance(multi, dict))

        assert (set(single) == {'Beta', 'P', 'Tau', 'S'})
        assert (set(multi).issubset(set(single)))

        self.Single = single
        self.Multi = multi
        self.test_ids_fn = test_ids_fn

    # creates all necessary data
    def run(self):
        for i in self.Multi:
            self._eval_param(i)
        self._eval_previous()
        self._eval_naive()

    # Decides which values to iterate
    def values(self, evaluated_parameter, param):
        if evaluated_parameter == 'all' or evaluated_parameter == param:
            return self.Multi[param]
        else:
            return self.Single[param]

    # Evaluations a single parameter
    def _eval_param(self, evaluated_parameter):
        print('Parsing parameter {} ... '.format(evaluated_parameter), end='', flush=True)
        fn = Name_functions.parameter_evaluation_evaluation_metric_file(evaluated_parameter)
        if Filefunctions.exists(fn):
            print('Already done')
            return

        with open(fn, 'w+') as wf:
            wf.write('S;Beta;Tau;P;Day;NumEntries;accuracy;f1\n')
            for S in self.values(evaluated_parameter, 'S'):
                predictor = Classifiers.BPTSClassifier(s=S, score_function=None)
                fn = Name_functions.DS_file(S)
                _, labels, times, ids = Di(fn).get_data(fn_subset_ids=self.test_ids_fn, return_split_values=True,
                                                        return_identifiers=True)
                data = pd.DataFrame(index=ids)
                data['time'] = times
                data['y_true'] = [l[0] for l in labels]
                data['Day'] = np.floor(data['time'])

                for beta in self.values(evaluated_parameter, 'Beta'):
                    for p in self.values(evaluated_parameter, 'P'):
                        for tau in self.values(evaluated_parameter, 'Tau'):
                            scoring_function = PeriodScoring(s=S, beta=beta, tau=tau, p=p)
                            predictor.set_scoring_function(scoring_function)
                            for day in data['Day'].unique():
                                subset = data[data['Day'] == day]
                                acc_score, f1_score = self.get_scores(predictor=predictor,
                                                                      ids=subset.index,
                                                                      times=subset['time'],
                                                                      true_labels=subset['y_true'],
                                                                      )
                                if not (acc_score is None or f1_score is None):
                                    wf.write('{};{};{};{};{};{};{};{}\n'.format(S,
                                                                                beta,
                                                                                tau,
                                                                                p,
                                                                                day,
                                                                                len(subset),
                                                                                acc_score,
                                                                                f1_score))
        print('Done')

    @staticmethod
    def get_scores(predictor, ids, times, true_labels):
        predicted_labels = predictor.predict_multiple(case_ids=ids, times=times)
        assert (len(true_labels) == len(predicted_labels))
        assert (isinstance(true_labels, pd.Series))
        assert (isinstance(predicted_labels, pd.Series))

        if true_labels.isna().all() or predicted_labels.isna().all():
            return None, None

        if true_labels.isna().any() or predicted_labels.isna().any():
            raise Exception('Help!')

        acc_score = Metrics.accuracy(true_label=true_labels,
                                     predicted_label=predicted_labels)
        f1_score = Metrics.f1(true_label=true_labels,
                              predicted_label=predicted_labels)
        return acc_score, f1_score

    # Evaluations the previous method
    def _eval_previous(self):
        print('Parsing Previous ... ', end='', flush=True)
        fn_recent = Name_functions.parameter_evaluation_evaluation_metric_file('Previous')
        if Filefunctions.exists(fn_recent):
            print('Already done')
            return

        with open(fn_recent, 'w+') as wf:
            wf.write('S;Day;NumEntries;accuracy;f1\n')
            for S in self.Multi['S']:
                predictor = Classifiers.PreviousClassifier(S)
                fn = Name_functions.DS_file(S)
                _, labels, times, ids = Di(fn).get_data(fn_subset_ids=self.test_ids_fn,
                                                        return_split_values=True,
                                                        return_identifiers=True)
                data = pd.DataFrame(index=ids)
                data['time'] = times
                data['y_true'] = [l[0] for l in labels]
                data['Day'] = np.floor(data['time'])

                # Calculate the accuracy score for each day
                for day in data['Day'].unique():
                    subset = data[data['Day'] == day]
                    acc_score, f1_score = self.get_scores(predictor=predictor,
                                                          true_labels=subset['y_true'],
                                                          times=subset['time'],
                                                          ids=subset.index
                                                          )
                    if not (acc_score is None or f1_score is None):
                        wf.write('{};{};{};{};{}\n'.format(S,
                                                           day,
                                                           len(subset),
                                                           acc_score,
                                                           f1_score))
        print('Done')

    # Evaluations the Naive method over time
    def _eval_naive(self):
        print('Parsing Naive ... ', end='', flush=True)
        fn_naive = Name_functions.parameter_evaluation_evaluation_metric_file('Naive')
        if Filefunctions.exists(fn_naive):
            print('Already done')
            return

        with open(fn_naive, 'w+') as wf:
            wf.write('S;Day;NumEntries;accuracy;f1\n')
            for S in self.Multi['S']:
                predictor = Classifiers.NaiveClassifier(S)
                fn = Name_functions.DS_file(S)
                _, labels, times, ids = Di(fn).get_data(fn_subset_ids=self.test_ids_fn,
                                                        return_split_values=True,
                                                        return_identifiers=True)
                data = pd.DataFrame(index=ids)
                data['time'] = times
                data['y_true'] = [l[0] for l in labels]
                data['Day'] = np.floor(data['time'])

                # Calculate the accuracy score for each day
                for day in data['Day'].unique():
                    if day < Parameters.train_time_naive_stop:
                        # We don't need this data
                        continue
                    subset = data[data['Day'] == day]
                    acc_score, f1_score = self.get_scores(predictor=predictor,
                                                          ids=subset.index,
                                                          times=subset['time'],
                                                          true_labels=subset['y_true']
                                                          )
                    if not (acc_score is None or f1_score is None):
                        wf.write('{};{};{};{};{}\n'.format(S,
                                                           day,
                                                           len(subset),
                                                           acc_score,
                                                           f1_score))
        print('Done')


# interface method
def run():
    with open(Name_functions.best_graec(), 'r') as rf:
        (S, B, T, P) = rf.readline()[:-1].split(';')[0:4]
    CalculateDailyScores(single={'S': [int(S)],
                                 'Tau': [float(T)],
                                 'P': [float(P)],
                                 'Beta': [float(B)]},
                         multi={'Beta': Parameters.GRAEC_beta,
                                'Tau': Parameters.GRAEC_tau,
                                'S': Parameters.S_values},
                         test_ids_fn=Name_functions.DS_reduced_ids_DSJ(S)).run()


def run_on():
    run()
    STEP9_Over_Time_Results.run_on()


def restart():
    undo()
    run_on()


def undo():
    Filefunctions.delete(Name_functions.parameter_evaluation_data_folder())
    STEP9_Over_Time_Results.undo()


if __name__ == '__main__':
    run()
