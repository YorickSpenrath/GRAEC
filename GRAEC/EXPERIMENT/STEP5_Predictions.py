from GRAEC.AUXILIARY_FUNCTIONS.Concept_Drift_Weighting_Functions import PeriodScoring
from GRAEC.AUXILIARY_FUNCTIONS.Data_Importer import DataImporter as DI
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions, Classifiers
from GRAEC import Parameters
from GRAEC.EXPERIMENT import STEP6_Global_Scoring
from GRAEC.Functions import Human_Functions, Filefunctions

test_min_start_time = Parameters.test_time_start
all_labels = Parameters.all_labels


def parse_ms(s):
    print('\tGRAEC ... ', end='', flush=True)
    if Filefunctions.exists(Name_functions.S_GRAEC_enumeration_dictionary(s)):
        print('Already done')
        return

    enumeration_encoder = dict()

    fn_data = Name_functions.DS_file(s)
    fn_train_ids = Name_functions.DS_train_ids(s)
    fn_test_ids = Name_functions.DS_test_ids(s)
    x_train, labels_train, times_train, ids_train = DI(fn_data).get_data(fn_subset_ids=fn_train_ids,
                                                                         return_split_values=True,
                                                                         return_identifiers=True)
    x_test, labels_test, times_test, ids_test = DI(fn_data).get_data(fn_subset_ids=fn_test_ids,
                                                                     return_split_values=True,
                                                                     return_identifiers=True)

    enumeration = 0
    predictor = Classifiers.BPTSClassifier(s=s, score_function=None)

    for B in Parameters.GRAEC_beta:
        for T in Parameters.GRAEC_tau:
            for P in Parameters.GRAEC_p if not T == 0 else [0]:  # P has no use for T == 0
                enumeration_encoder[enumeration] = '{};{};{}'.format(B, T, P)
                predictor.set_scoring_function(score_function=PeriodScoring(beta=B, p=P, tau=T, s=s))

                with open(Name_functions.S_GRAEC_train_predictions(s, enumeration), 'w+') as wf:
                    wf.write('SOID;time;True_label;Predicted_label\n')
                    for case_id, t, true_label in zip(ids_train, times_train, labels_train):
                        predicted_label = predictor.predict(case_id=case_id, time=t)
                        wf.write('{};{};{};{}\n'.format(case_id, t, true_label[0], predicted_label))

                with open(Name_functions.S_GRAEC_test_predictions(s, enumeration), 'w+') as wf:
                    wf.write('Case_id;time;True_label;Predicted_label\n')
                    for case_id, t, true_label in zip(ids_test, times_test, labels_test):
                        predicted_label = predictor.predict(case_id=case_id, time=t)
                        wf.write('{};{};{};{}\n'.format(case_id, t, true_label[0], predicted_label))

                enumeration += 1

    Human_Functions.save_dict_to_csv(enumeration_encoder, Name_functions.S_GRAEC_enumeration_dictionary(s))

    fn_data = Name_functions.DS_file(s)
    fn_ids = Name_functions.DS_test_ids(s)
    x, labels, times, ids = DI(fn_data).get_data(fn_subset_ids=fn_ids,
                                                 return_split_values=True,
                                                 return_identifiers=True)

    print('Done')
    print('\tNaive and Previous ... ', end='', flush=True)

    naive_predictor = Classifiers.NaiveClassifier(s)
    previous_predictor = Classifiers.PreviousClassifier(s)
    with open(Name_functions.S_naive_test_predictions(s), 'w+') as wf_naive:
        with open(Name_functions.S_recent_test_predictions(s), 'w+') as wf_previous:
            wf_naive.write('{};{};{};{}\n'.format('Case_id', 'time', 'True_label', 'Predicted_label'))
            wf_previous.write('{};{};{};{}\n'.format('Case_id', 'time', 'True_label', 'Predicted_label'))
            for case_id, t, true_label in zip(ids, times, labels):
                predicted_label_naive = naive_predictor.predict(case_id=case_id, time=t)
                if predicted_label_naive is not None:
                    wf_naive.write('{};{};{};{}\n'.format(case_id, t, true_label[0], predicted_label_naive))
                predicted_label_previous = previous_predictor.predict(case_id=case_id, time=t)
                if predicted_label_previous is not None:
                    wf_previous.write('{};{};{};{}\n'.format(case_id, t, true_label[0], predicted_label_previous))
    print('Done')


def run():
    for S in Parameters.S_values:
        print('S = {}'.format(S))
        parse_ms(S)


def run_on():
    run()
    STEP6_Global_Scoring.run_on()


def restart():
    undo()
    run_on()


def undo():
    for S in Parameters.S_values:
        Filefunctions.delete(Name_functions.S_GRAEC_enumeration_folder(S))
        Filefunctions.delete(Name_functions.S_naive_test_predictions(S))
        Filefunctions.delete(Name_functions.S_recent_test_predictions(S))
    STEP6_Global_Scoring.undo()


if __name__ == '__main__':
    run()
