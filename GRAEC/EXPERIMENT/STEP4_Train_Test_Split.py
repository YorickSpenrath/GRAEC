"""
Splits the test wp_dataset into a set for training the Proposed method parameters and a set for testing the accuracy of
the three (Naive, Recent, GRAEC) methods
"""

from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter as DI
from GRAEC.Functions import Filefunctions
import numpy as np


def parse_ms(s):
    print('D^{} ... '.format(s), end='', flush=True)
    if Filefunctions.exists(Name_functions.DS_train_ids(s)):
        if Filefunctions.exists(Name_functions.DS_test_ids(s)):
            print('Already done')
            return

    np.random.seed(0)
    X, y, times, ids = DI(Name_functions.DS_file(s)).get_data(
        Name_functions.DS_reduced_ids_DSJ(s), True, True)

    if Parameters.take_test_split_chronological:
        test_case_ids = []
        train_case_ids = []
        times_post_warm_up = [t for t in times if t > Parameters.test_time_start]
        times_post_warm_up.sort()
        train_start_index = int((1 - Parameters.assessment_test_split) * len(times_post_warm_up))
        train_time_end = times_post_warm_up[train_start_index]
        for case_start_time, case_id in zip(times, ids):
            if case_start_time <= Parameters.test_time_start:
                continue

            if case_start_time < train_time_end:
                train_case_ids.append(case_id)
            else:
                test_case_ids.append(case_id)
    else:
        indices = [i for i in range(len(ids)) if times[i] > Parameters.test_time_start]
        test_indices = []
        train_indices = []
        c, cc = np.unique(y[indices], return_counts=True)
        for label, label_count in zip(c, cc):
            num_test = int(label_count * Parameters.assessment_test_split)
            indices_c = [i for i in indices if y[i] == label]
            indices_c_test = np.random.choice(indices_c, num_test, replace=False)
            test_indices.extend(indices_c_test.tolist())
            train_indices.extend([i for i in indices_c if i not in indices_c_test])
        test_case_ids = ids[test_indices]
        train_case_ids = ids[train_indices]

    with open(Name_functions.DS_train_ids(s), 'w+') as wf:
        for case_id in train_case_ids:
            wf.write('{}\n'.format(case_id))

    with open(Name_functions.DS_test_ids(s), 'w+') as wf:
        for case_id in test_case_ids:
            wf.write('{}\n'.format(case_id))

    print('Done')


def run():
    for S in Parameters.S_values:
        parse_ms(S)


if __name__ == '__main__':
    run()
