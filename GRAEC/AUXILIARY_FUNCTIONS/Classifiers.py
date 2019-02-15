"""
This file handles the three different types of classifiers used in the experiment (Naive, Recent, and GRAEC)
"""

import abc

from GRAEC.AUXILIARY_FUNCTIONS.Concept_Drift_Weighting_Functions import BEPTScoring, PeriodScoring
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
import pandas as pd
import numpy as np


class BaseClassifier:

    def __init__(self):
        pass

    def predict(self, case_id, time):
        probabilities = self.predict_probabilities(case_id=case_id, time=time)
        if probabilities is None:
            return None
        if sum(probabilities) == 0:
            return None
        return Parameters.all_labels[np.argmax(probabilities)]

    @abc.abstractmethod
    def predict_probabilities(self, case_id, time):
        pass

    def predict_multiple(self, case_ids, times):
        ids = []
        labels = []
        for case_id, t in zip(case_ids, times):
            ids.append(case_id)
            labels.append(self.predict(case_id, t))
        return pd.Series(data=labels, index=ids)


class CDClassifier(BaseClassifier):

    def __init__(self, s, score_function):
        super().__init__()
        assert (isinstance(score_function, BEPTScoring) or score_function is None)
        self.score_function = score_function
        self.predictions = Name_functions.import_probabilities_split(S=s)
        self.times = Name_functions.SJ_period_mid_times(S=s)

    def predict_probabilities(self, case_id, time):
        if case_id not in self.predictions:
            return None

        probabilities = np.array([0.0] * len(Parameters.all_labels))
        for i in self.predictions[case_id]:
            assert (self.times[i] < time)
            probabilities += np.array(self.predictions[case_id][i]) * self.score_function.weight(time - self.times[i])

        return probabilities

    def set_scoring_function(self, score_function):
        assert (isinstance(score_function, BEPTScoring))
        self.score_function = score_function


class BPTSClassifier(BaseClassifier):
    def __init__(self, s, score_function):
        super().__init__()
        assert (score_function is None or isinstance(score_function, PeriodScoring))
        self.score_function = score_function
        self.predictions = Name_functions.import_probabilities_split(S=s)

    def predict_probabilities(self, case_id, time):
        if case_id not in self.predictions:
            return None

        probabilities = np.array([0.0] * len(Parameters.all_labels))
        for period, weight in self.score_function.get_weights(data_time=time).items():
            if str(period) in self.predictions[case_id]:
                probabilities += np.array(self.predictions[case_id][str(period)]) * weight

        return probabilities

    def set_scoring_function(self, score_function):
        assert (isinstance(score_function, PeriodScoring))
        self.score_function = score_function


class PreviousClassifier(BaseClassifier):

    def __init__(self, s):
        super().__init__()
        self.predictions = Name_functions.import_probabilities_split(S=s)
        self.times = Name_functions.SJ_period_end_times(S=s)

    def predict_probabilities(self, case_id, time):
        if case_id not in self.predictions:
            return None

        available_models = [i for i in self.predictions[case_id] if (self.times[i] < time)]
        if len(available_models) == 0:
            return None

        return self.predictions[case_id][max(available_models, key=lambda i: float(i))]


class NaiveClassifier(BaseClassifier):
    def __init__(self, s):
        super().__init__()
        self.predictions = Name_functions.import_probabilities_naive(S=s)

    def predict_probabilities(self, case_id, time):
        if case_id not in self.predictions:
            return None
        else:
            return self.predictions[case_id]
