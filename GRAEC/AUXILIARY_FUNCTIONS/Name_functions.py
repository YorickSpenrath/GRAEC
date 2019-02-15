"""
This file handles all the complications involved in the file structure of splitting data in different ways.
"""
from GRAEC.Functions import Filefunctions
from GRAEC import Parameters
import os
import numpy as np


########################################################################################################################
# # File structure
# Root of data
def root():
    return Parameters.root_location


# Root of moment + S
def S_folder(S):
    return root() + '/Datasets/{}'.format(S)


# Root folder of a split
def SJ_folder(S, i):
    return S_folder(S) + '/Splits/{}'.format(i)


# periods we consider
def S_J_values(S):
    folder = S_folder(S) + '/Splits'
    if Filefunctions.exists(folder):
        return os.listdir(folder)
    else:
        return []


# Root of the enumeration files
def S_GRAEC_enumeration_folder(S):
    fd = S_folder(S) + '/BEPT_Enumeration'
    Filefunctions.make_directory(fd)
    return fd


########################################################################################################################
def results_folder():
    return root() + '/Results'


def metric_figure(metric):
    fn = results_folder() + '/Result_{}.png'.format(metric)
    Filefunctions.makeParentDir(fn)
    return fn


def metric_table(metric):
    fn = results_folder() + '/GRAEC TexTable {}.txt'.format(metric)
    Filefunctions.makeParentDir(fn)
    return fn


def full_GRAEC_table():
    fn = results_folder() + '/GRAEC Full Scores.csv'
    Filefunctions.makeParentDir(fn)
    return fn


########################################################################################################################
# # Data
# Raw data
def DS_file(S):
    return S_folder(S) + '/Data.csv'


# CaseID's of the reduced data (based on k medoid reduction per period)
def DS_reduced_ids_DSJ(S):
    return S_folder(S) + '/Periodic_Reducted_IDS.csv'


# CaseID's of the reduced data (based on k medoid reduction for Naive training period)
def DS_reduced_ids_naive(S):
    return S_folder(S) + '/Naive_Reducted_IDS.csv'


# File with the IDS that are used in the evaluation
def DS_test_ids(S):
    return S_folder(S) + '/Test_IDS.csv'


# File with the IDS that are used in optimising the BEPT parameters
def DS_train_ids(S):
    return S_folder(S) + '/Train_IDS.csv'


########################################################################################################################
# PARAMETER_EVALUATION

# Folder for the parameter evaluation experiment
def parameter_evaluation_folder():
    return root() + '/Parameter_Evaluation'.format()


# File for the best graec score
def best_graec():
    fn = parameter_evaluation_folder() + '/best_graec_parameters.csv'
    Filefunctions.makeParentDir(fn)
    return fn


def parameter_evaluation_data_folder():
    return parameter_evaluation_folder() + '/Data'


# File for the evaluation of a given parameter
def parameter_evaluation_evaluation_metric_file(parameter):
    fn = parameter_evaluation_data_folder() + '/{}.csv'.format(parameter)
    Filefunctions.makeParentDir(fn)
    return fn


def parameter_metric_evaluation_folder(metric):
    return parameter_evaluation_folder() + '/{}'.format(metric)


def parameter_evaluation_figure(parameter, metric):
    fn = parameter_metric_evaluation_folder(metric) + '/{}.png'.format(parameter)
    Filefunctions.makeParentDir(fn)
    return fn


########################################################################################################################
# Filename of the predictions probabilities made by the Naive model
def DS_probabilities_naive(S):
    return S_folder(S) + '/Naive_Probabilities.csv'


# File names of the predictions probabilities made by each split
def DSJ_probabilities(S, i):
    return SJ_folder(S, i) + '/Predictions.csv'


# get all predictions from all splits
# structured in a dict of dicts, where the top level keys are case ids, and the second level keys are splits
# mapping to predictions
def import_probabilities_split(S):
    ret = dict()
    for i in S_J_values(S):
        with open(DSJ_probabilities(S, i), 'r') as rf:
            for line in rf.readlines():
                line_split = line[:-1].split(';')
                ret.setdefault(line_split[0], dict())[i] = np.array([float(j) for j in line_split[2:]])
    return ret


# Import all predictions made by the naive model. Single dict mapping caseID to predictions
def import_probabilities_naive(S):
    ret = dict()
    fn_predictions = DS_probabilities_naive(S)
    with open(fn_predictions, 'r') as rf:
        for line in rf.readlines():
            line_split = line[:-1].split(';')
            ret[line_split[0]] = np.array([float(j) for j in line_split[2:]])
    return ret


########################################################################################################################
# # Labels
# Filename of the predictions made by the Naive model
def S_naive_test_predictions(S):
    return S_folder(S) + '/Naive_Test_Predictions.csv'


def S_recent_test_predictions(S):
    return S_folder(S) + '/Previous_Test_Predictions.csv'


########################################################################################################################
# Score files

def S_score_folder(S):
    return S_folder(S) + '/scores'


# Filename of the score of the Naive model
def S_naive_score(S, metric):
    return S_score_folder(S) + '/NAIVE_{}.csv'.format(metric)


# Filename of the score of the BEPT method
def S_GRAEC_score(S, metric):
    return S_score_folder(S) + '/BEPT_{}.csv'.format(metric)


# File with the score of the Recent method
def S_recent_score(S, metric):
    return S_score_folder(S) + '/PREVIOUS_{}.csv'.format(metric)


########################################################################################################################
# # Models
# Filename of the Naive model
def model_S_naive(S):
    return S_folder(S) + '/Naive_Model.model'


# model name of a split
def model_SJ(S, i):
    return SJ_folder(S, i) + '/Model.model'.format(i)


########################################################################################################################
# # BEPT Enumeration
# Encoding of enumerations
def S_GRAEC_enumeration_dictionary(S):
    return S_GRAEC_enumeration_folder(S) + '/Encoder.csv'


# Predictions made by each enumeration
def S_GRAEC_train_predictions(S, enumeration):
    return S_GRAEC_enumeration_folder(S) + '/{} Training.csv'.format(enumeration)


# Predictions made by each enumeration
def S_GRAEC_test_predictions(S, enumeration):
    return S_GRAEC_enumeration_folder(S) + '/{} Test.csv'.format(enumeration)


########################################################################################################################
# # Time values of splits
# Get the time representation of each model
def SJ_period_mid_times(S):
    return {i: ((2 * int(i) + 1) * int(S)) / 2 for i in S_J_values(S)}


# Get the end time of each model
def SJ_period_end_times(S):
    return {i: (int(i) + 1) * int(S) for i in S_J_values(S)}


# Get the end time of a specific model
def SJ_period_end_time(S, i):
    return (int(i) + 1) * int(S)


# Get the start time of each model
def SJ_period_start_times(S):
    return {i: (int(i)) * int(S) for i in S_J_values(S)}


########################################################################################################################
def event_log():
    if Parameters.Demo:
        return root() + '/Event_Log.csv'
    else:
        raise Exception('Not in Demo mode')


def cases_info():
    if Parameters.Demo:
        return root() + '/Cases_Info.csv'
    else:
        raise Exception('Not in Demo mode')
