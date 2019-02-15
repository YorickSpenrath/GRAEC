"""
Trains all the required models
"""
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter as Di
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters

from GRAEC.Functions import Model_Functions
import numpy as np

import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def parse_naive(s):
    print('\tM^{}_naive ... '.format(s), end='', flush=True)
    fn_model = Name_functions.model_S_naive(s)
    if os.path.exists(fn_model):
        print("Already done")
        return 1.0, 1.0, 100

    fn_data = Name_functions.DS_file(s)
    fn_subset = Name_functions.DS_reduced_ids_DSJ(s)

    x, y, t = Di(fn_data).get_data(fn_subset, True, False)

    y = y.ravel()

    # Only take data that is in the first year
    x = [x[i] for i in range(len(t)) if t[i] < Parameters.train_time_naive_stop]
    y = [y[i] for i in range(len(t)) if t[i] < Parameters.train_time_naive_stop]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

    # Get the best model
    best_model = None
    best_score = -1
    for c in used_models:
        score, model = train_classifier(c, x_train, x_test, y_train, y_test)
        if score > best_score:
            best_score = score
            best_model = model

    # save the model
    Model_Functions.saveModel(best_model, fn_model)
    print("Done")
    return 1.0, 1.0, 100


def parse_s(s):
    fn_data = Name_functions.DS_file(s)
    fn_subset = Name_functions.DS_reduced_ids_DSJ(s)
    x, y = Di(fn_data).split_data(int(s), fn_subset_ids=fn_subset)
    print('\tM^{}_j ... '.format(s), end='', flush=True)
    good_splits = 0

    for i in sorted(x):
        fn_model = Name_functions.model_SJ(s, i)
        c, cc = np.unique(y[i], return_counts=True)

        if min(cc) < cv * 2:
            index = np.where(cc == np.min(cc))
            continue
        if len(c) <= 1:
            continue
        if os.path.exists(fn_model):
            good_splits += 1
            continue
        else:
            generate_model(x[i], y[i], s, i)
            good_splits += 1
            continue
    print('Done ({}/{} D^{}_j met requirements)'.format(good_splits, len(x), s))
    return good_splits, len(x), 100 * good_splits / len(x)


def generate_model(x, y, s, i):

    # Generate Train/Test
    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), random_state=0, test_size=0.2)

    # Get the best model
    best_model = None
    best_score = -1
    for c in used_models:
        score, model = train_classifier(c, x_train, x_test, y_train, y_test)
        if score > best_score:
            best_score = score
            best_model = model

    # save the model
    Model_Functions.saveModel(best_model, Name_functions.model_SJ(s, i))


def train_classifier(classifier, x_train, x_test, y_train, y_test):
    hyper_parameter_grid = Model_Functions.get_HP(classifier)
    use_grid_search = force_gridSearch or np.prod(
        np.array([len(hyper_parameter_grid[n]) for n in hyper_parameter_grid])) <= max_iter

    # setup grid
    if use_grid_search:
        grid = GridSearchCV(
            estimator=Model_Functions.getModel(classifier),
            param_grid=hyper_parameter_grid,
            cv=cv,
            refit=True
        )
    else:
        grid = RandomizedSearchCV(
            estimator=Model_Functions.getModel(classifier),
            param_distributions=hyper_parameter_grid,
            n_iter=max_iter,
            cv=cv,
            random_state=0,
            refit=True
        )
    grid.fit(x_train, y_train)
    return grid.score(x_test, y_test), grid.best_estimator_


force_gridSearch = Parameters.force_gridSearch
max_iter = Parameters.max_iter
cv = Parameters.cv
used_models = Parameters.used_models


def run():
    for S in Parameters.S_values:
        print('S = {}'.format(S))
        parse_s(S)
        parse_naive(S)


if __name__ == '__main__':
    run()
