'''
Created Date: Tuesday November 27th 2018
Last Modified: Tuesday November 27th 2018 10:44:19 pm
Author: ankurrc
'''

from sklearn.metrics import f1_score, precision_score, recall_score


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred)
