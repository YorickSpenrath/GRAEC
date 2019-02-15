from GRAEC.Functions import Filefunctions


def load_dict_from_csv(filename):
    assert (Filefunctions.exists(filename))
    ret = dict()
    with open(filename, 'r') as rf:
        for line in rf.readlines():
            k, v = line[:-1].split(';', 1)
            ret[k] = v
    return ret


def save_dict_to_csv(dictionary, filename):
    assert (isinstance(dictionary, dict))
    Filefunctions.makeParentDir(filename)
    with open(filename, 'w+') as wf:
        for k, v in dictionary.items():
            wf.write('{};{}\n'.format(k, v))

