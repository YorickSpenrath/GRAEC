"""
Reduces the number of data points in each subset, by taking a stratified K-medoids to ensure the largest class is at
most the given factor larger than the smallest class
"""

from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter
from GRAEC.Functions import Filefunctions, KMedoids


def parse(s):
    parse_ms(s)
    parse_naive(s)


def parse_ms(s):
    fn_target = Name_functions.DS_reduced_ids_DSJ(s)

    # Check existence
    print('\tD^S_j ... ', end='', flush=True)
    if Filefunctions.exists(fn_target):
        print('Already done')
        return

    fn_input = Name_functions.DS_file(s)
    x, y, ids = DataImporter(fn_input).split_data(int(s), return_identifiers=True)
    ids_keep = []
    for i in sorted(x):
        xi, yi, indices = KMedoids.reduce_to_medoids(x[i], y[i], return_indices=True)
        ids_keep.extend([ids[i][j] for j in indices])

    with open(fn_target, 'w+') as wf:
        for caseID in ids_keep:
            wf.write('{}\n'.format(caseID))

    print('Done')


def parse_naive(s):
    fn_target = Name_functions.DS_reduced_ids_naive(s)

    print('\tD^S_naive ... ', end='', flush=True)
    # Check existence
    if Filefunctions.exists(fn_target):
        print('Already done')
        return

    fn_input = Name_functions.DS_file(s)
    x, y, timestamps, ids = DataImporter(fn_input).get_data(return_identifiers=True, return_split_values=True)

    first_year_indices = [i for i in range(len(timestamps)) if timestamps[i] < Parameters.train_time_naive_stop]
    x = x[first_year_indices]
    y = y[first_year_indices]
    ids = ids[first_year_indices]
    x, y, medoid_indices = KMedoids.reduce_to_medoids(x, y, return_indices=True, factor=Parameters.LargeSmallFactor)
    ids_keep = [ids[i] for i in medoid_indices]

    with open(fn_target, 'w+') as wf:
        for CaseID in ids_keep:
            wf.write('{}\n'.format(CaseID))

    print('Done')


def run():
    for S in Parameters.S_values:
        print('S = {}'.format(S))
        parse(S)


if __name__ == '__main__':
    run()
