"""
This script saves all predictions made by a model on the test data, to speed up the remaining steps by not having to run
a model many times
"""
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter as Di
from GRAEC.Functions import Filefunctions, Model_Functions

all_labels = Parameters.all_labels


def parse_ms(s):
    fn_data = Name_functions.DS_file(s)
    x, y, time, case_id = Di(fn_data).get_data(return_identifiers=True, return_split_values=True)

    print('\tM^{}_j ... '.format(s), end='', flush=True)

    # S predictions
    for i in sorted([int(i) for i in Name_functions.S_J_values(s)], reverse=True):

        if Filefunctions.exists(Name_functions.DSJ_probabilities(s, i)):
            continue

        model_i = Model_Functions.loadModel(Name_functions.model_SJ(s, i))
        model_labels = model_i.classes_.tolist()
        model_end_time = Name_functions.SJ_period_end_time(s, i)

        with open(Name_functions.DSJ_probabilities(s, i), 'w+') as wf:
            for dx, t, idn in zip(x, time, case_id):
                if t < model_end_time:
                    # Only test if the model existed before the data point
                    continue
                model_predictions = model_i.predict_proba(dx.reshape(1, -1))[0]
                actual_predictions = [(0 if (i not in model_labels) else model_predictions[model_labels.index(i)]) for i
                                      in all_labels]
                wf.write('{};{};{}\n'.format(idn, t, ';'.join(['{:4f}'.format(x) for x in actual_predictions])))
    print('Done')

    # Naive predictions
    print('\tM^{}_naive ... '.format(s), end='', flush=True)
    if Filefunctions.exists(Name_functions.DS_probabilities_naive(s)):
        print('Already done')
        return

    model_naive = Model_Functions.loadModel(Name_functions.model_S_naive(s))
    model_naive_labels = model_naive.classes_.tolist()
    model_naive_end_time = Parameters.train_time_naive_stop

    with open(Name_functions.DS_probabilities_naive(s), 'w+') as wf:
        for dx, t, idn in zip(x, time, case_id):
            if t < model_naive_end_time:
                # Only test if the model existed before the data point
                continue

            model_predictions = model_naive.predict_proba(dx.reshape(1, -1))[0]
            actual_predictions = [
                (0 if (i not in model_naive_labels)
                 else model_predictions[model_naive_labels.index(i)])
                for i in all_labels]
            wf.write('{};{};{}\n'.format(idn, t, ';'.join(['{:4f}'.format(x) for x in actual_predictions])))
    print('Done')


def run():
    for S in Parameters.S_values:
        print('S = {}'.format(S))
        parse_ms(int(S))


if __name__ == '__main__':
    run()
