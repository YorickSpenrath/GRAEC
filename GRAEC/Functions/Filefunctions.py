"""
Created on Wed Oct 17 09:30:34 2018

@author: 179040

Graduation Project Yorick Spenrath
Eindhoven University of Technology
For the degree "Master of Science"
In the programs Business Information Systems
                Operations Management and Logistics
In association with Kropman B.V., Nijmegen
For more info please contact the original author at "yorick.spenrath@gmail.com"

"""
import pickle

"""
Functions to handle remove and creation of files
"""

import shutil
import os


def str_assert(string):
    if not isinstance(string, str):
        raise Exception('Not a string:\n{}'.format(str))


def exists_assert(filename):
    assert exists(filename), 'File does not exist:\n{}'.format(filename)


def clear_directory(fd):
    str_assert(fd)
    delete(fd)
    make_directory(fd)


def make_directory(fd):
    str_assert(fd)
    if not os.path.exists(fd):
        os.makedirs(fd)


def copyfile(src, des):
    str_assert(src)
    str_assert(des)
    assert (exists(src))
    makeParentDir(des)
    shutil.copyfile(src, des)


def delete(filename):
    str_assert(filename)
    if not exists(filename):
        return
    if os.path.isdir(filename):
        shutil.rmtree(filename)
    else:
        os.remove(filename)


def exists(filename):
    str_assert(filename)
    return os.path.exists(filename)


def makeParentDir(filename):
    str_assert(filename)
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def list_files(file_folder, as_paths=True):
    str_assert(file_folder)
    assert (exists(file_folder))
    return [(file_folder + '/' + f if as_paths else f) for f in os.listdir(file_folder) if
            os.path.isfile(os.path.join(file_folder, f))]


def save_obj(obj, filename):
    str_assert(filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    str_assert(filename)
    with open(filename, 'rb') as f:
        return pickle.load(f)
