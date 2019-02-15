"""
Wrapper code for sk learn to ensure global settings for each metric are the same
"""

from sklearn.metrics import accuracy_score, f1_score


def accuracy(true_label, predicted_label):
    return accuracy_score(true_label, predicted_label)


def f1(true_label, predicted_label):
    return f1_score(y_true=true_label, y_pred=predicted_label, average='weighted')
