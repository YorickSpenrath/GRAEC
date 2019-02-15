from GRAEC.Functions import Filefunctions
from GRAEC.AUXILIARY_FUNCTIONS import Name_functions
from GRAEC import Parameters
from GRAEC.AUXILIARY_FUNCTIONS.Bottleneck_Algorithm import algorithm
from GRAEC.AUXILIARY_FUNCTIONS.Process_Units import EventLog


def create_labelled_dataset(event_log, s, feature_filename, output_file):
    # Check or create Event Log
    assert (isinstance(s, int))
    if isinstance(event_log, EventLog):
        pass
    elif isinstance(event_log, str):
        event_log = EventLog(filename=event_log)
    else:
        raise Exception('Given argument should be filename or EventLog')

    # Dictionary with all labels
    labels = dict()

    # Each split of the full dataset gets parsed separately
    splits = event_log.get_splits(s)
    for split in splits.values():
        labels.update(algorithm(split))

    # Create labelled data
    with open(feature_filename, 'r') as rf:
        Filefunctions.makeParentDir(output_file)

        with open(output_file, 'w+') as wf:
            wf.write(rf.readline()[:-1] + ';Case_start;Class\n')
            wf.write(rf.readline()[:-1] + ';-;-\n')
            wf.write(rf.readline()[:-1] + ';-;-\n')
            wf.write(rf.readline()[:-1] + ';SPLIT;Y\n')
            for line in rf.readlines():
                values = line[:-1].split(';')
                case = event_log.get_case(values[0])
                values.append(str(case.get_start()))
                values.append(labels[values[0]])
                wf.write(';'.join(values) + '\n')


def run():
    for s in Parameters.S_values:
        create_labelled_dataset(event_log=Name_functions.event_log(),
                                s=s,
                                feature_filename=Name_functions.cases_info(),
                                output_file=Name_functions.DS_file(s))
