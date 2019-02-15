"""
Created on Wed Oct 17 11:57:43 2018

@author: 179040

Graduation Project Yorick Spenrath
Eindhoven University of Technology
For the degree "Master of Science"
In the programs Business Information Systems
                Operations Management and Logistics
In association with Kropman B.V., Nijmegen
For more info please contact the original author at "yorick.spenrath@gmail.com"

"""

"""
Functions relating to model settings
"""
import pickle
import os

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


def saveModel(model, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as model_saver:
        pickle.dump(model, model_saver)
    return


def loadModel(filename):
    return pickle.load(open(filename, 'rb'))


def get_Models(name):
    if name == 'all_classifiers':
        return ['Logistic_Regression', 'Nearest_Neighbors', 'SVC', 'Neural_Network', 'Decision_Tree', 'Random_Forest', ]
    if name == 'NN_only':
        return ['Neural_Network', ]
    if name == 'SCV_ony':
        return ['SVC']
    if name == 'fast_classifiers':
        return ['Logistic_Regression', 'Nearest_Neighbors', 'Decision_Tree', 'Random_Forest', ]
    if name == 'superspeed_classifiers':
        return ['Logistic_Regression', ]
    if name == 'Demo':
        return ['Demo_DT']
    raise Exception("input not known")


def get_HP(classifier):
    LR_params = {'C': [10 ** x for x in range(-5, 5)], }
    kNN_params = {'n_neighbors': np.arange(1, 10).tolist(),
                  'weights': ['uniform', 'distance'], }
    SVC_params = {'C': [10 ** x for x in range(-5, 5)],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], }
    DT_params = {'max_depth': (5 * np.arange(1, 4)).tolist(),
                 'max_features': ["auto", "log2", 1.0], }
    RF_params = {'n_estimators': (5 * np.arange(1, 11)).tolist(),
                 'max_features': ["auto", "log2", 1.0],
                 'max_depth': (5 * np.arange(1, 4)).tolist(), }
    AB_params = {'n_estimators': (5 * np.arange(1, 11)).tolist(), }
    NN_params = {'hidden_layer_sizes': [(int(100 / x),) * x for x in np.arange(1, 11).tolist()],
                 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                 'alpha': [10 ** x for x in range(-5, 5)]}
    DDT_params = {'max_depth': [3],
                  'max_features': [1.0]}

    Params = {'Logistic_Regression': LR_params,
              'Nearest_Neighbors': kNN_params,
              'SVC': SVC_params,
              'Decision_Tree': DT_params,
              'Random_Forest': RF_params,
              'AdaBoost': AB_params,
              'Neural_Network': NN_params,
              'Demo_DT': DDT_params
              }

    return Params[classifier]


def getModel(classifier):
    all = {'Logistic_Regression': LogisticRegression(),
           'Nearest_Neighbors': KNeighborsClassifier(),
           'SVC': SVC(),
           'Neural_Network': MLPClassifier(max_iter=500),
           'Decision_Tree': DecisionTreeClassifier(),
           'Random_Forest': RandomForestClassifier(),
           'AdaBoost': AdaBoostClassifier(),
           'Demo_DT': DecisionTreeClassifier(),
           }
    return all[classifier]


def cv():
    return 5


def LargeSmallFactor():
    return 2


def max_iter():
    return 20


def force_gridSearch():
    return False


def dfun():
    return 'jaccard'
