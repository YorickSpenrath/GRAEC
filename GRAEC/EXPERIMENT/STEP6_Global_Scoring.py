"""
Handles the scoring of each of the 12 methods
"""
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions, Metrics
from GRAEC import Parameters
from GRAEC.EXPERIMENT import STEP7_Global_Results
from GRAEC.Functions import Human_Functions, DataFrameOperations, Filefunctions

all_labels = Parameters.all_labels


def parse_graec(s):
    # do GRAEC SCORE
    enumeration_encoder = Human_Functions.load_dict_from_csv(Name_functions.S_GRAEC_enumeration_dictionary(s))

    best_acc_train = -1
    best_acc_test = -1
    best_acc_enum = -1
    best_f1_train = -1
    best_f1_test = -1
    best_f1_enum = -1

    with open(Name_functions.full_GRAEC_table(), 'a+') as wf:

        for e in enumeration_encoder:
            df_train = DataFrameOperations.import_df(fn=Name_functions.S_GRAEC_train_predictions(s, e))
            acc_train = Metrics.accuracy(df_train['True_label'], df_train['Predicted_label'])
            f1_train = Metrics.f1(df_train['True_label'], df_train['Predicted_label'])

            df_test = DataFrameOperations.import_df(fn=Name_functions.S_GRAEC_test_predictions(s, e))
            acc_test = Metrics.accuracy(df_test['True_label'], df_test['Predicted_label'])
            f1_test = Metrics.f1(df_test['True_label'], df_test['Predicted_label'])

            wf.write(enumeration_encoder[e] + ';{};ACC;{}\n'.format(s, acc_test))
            wf.write(enumeration_encoder[e] + ';{};F1;{}\n'.format(s, f1_test))

            if acc_train > best_acc_train:
                best_acc_train = acc_train
                best_acc_test = acc_test
                best_acc_enum = e

            if f1_train > best_f1_train:
                best_f1_train = f1_train
                best_f1_test = f1_test
                best_f1_enum = e

    with open(Name_functions.S_GRAEC_score(s, 'accuracy'), 'w+') as wf:
        wf.write(enumeration_encoder[best_acc_enum] + '\n')
        wf.write('{}\n'.format(best_acc_test))

    with open(Name_functions.S_GRAEC_score(s, 'f1'), 'w+') as wf:
        wf.write(enumeration_encoder[best_f1_enum] + '\n')
        wf.write('{}\n'.format(best_f1_test))


def parse_previous(s):
    # do PREVIOUS SCORE
    df_previous = DataFrameOperations.import_df(fn=Name_functions.S_recent_test_predictions(s))
    predicted_label = df_previous['Predicted_label']
    true_label = df_previous['True_label']

    acc_score = Metrics.accuracy(true_label=true_label, predicted_label=predicted_label)
    with open(Name_functions.S_recent_score(s, 'accuracy'), 'w+') as wf:
        wf.write('{}\n'.format(acc_score))

    f1_score = Metrics.f1(true_label=true_label, predicted_label=predicted_label)
    with open(Name_functions.S_recent_score(s, 'f1'), 'w+') as wf:
        wf.write('{}\n'.format(f1_score))


def parse_naive(s):
    # do NAIVE SCORE
    df_naive = DataFrameOperations.import_df(fn=Name_functions.S_naive_test_predictions(s))
    predicted_label = df_naive['Predicted_label']
    true_label = df_naive['True_label']

    acc_score = Metrics.accuracy(true_label=true_label, predicted_label=predicted_label)
    with open(Name_functions.S_naive_score(s, 'accuracy'), 'w+') as wf:
        wf.write('{}\n'.format(acc_score))

    f1_score = Metrics.f1(true_label=true_label, predicted_label=predicted_label)
    with open(Name_functions.S_naive_score(s, 'f1'), 'w+') as wf:
        wf.write('{}\n'.format(f1_score))


def run():
    fn = Name_functions.full_GRAEC_table()

    # Clear Test Scores file
    with open(fn, 'w+') as wf:
        wf.write('Beta;Tau;P;S;Score_type;Score_Value\n')

    for S in Parameters.S_values:
        Filefunctions.make_directory(Name_functions.S_score_folder(S))
        print('S = {}'.format(S))
        parse_graec(S)
        parse_previous(S)
        parse_naive(S)


def run_on():
    run()
    STEP7_Global_Results.run_on()


def restart():
    undo()
    run_on()


def undo():
    for S in Parameters.S_values:
        Filefunctions.delete(Name_functions.S_score_folder(S))
    STEP7_Global_Results.undo()


if __name__ == '__main__':
    run()
