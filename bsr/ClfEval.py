from __future__ import print_function

from bsrdata import Sample, Spectrogram, Template

import logging

from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from SampleRepository import *
from operator import add

from sklearn.model_selection import ParameterGrid
from itertools import islice
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import resource

import copy

import gc
import time
import numpy as np
import math
import multiprocessing as mp
from collections import defaultdict

from sklearn.model_selection import train_test_split
from IPython.core.debugger import Tracer
from optparse import OptionParser
import cv2
import random

class ClfEval:
    clf = None

    cnf = None

    feature_importances = None

    stats = None

    n_shuffles=0
    n_splits=0

    data=None

    random_state=None

    reject_templates = None


    def __init__(self, data, n_shuffles=1, n_splits=0, random_state=None):
        self.n_shuffles = n_shuffles
        self.n_splits = n_splits
        self.data = data
        self.random_state = random_state
        self.reject_templates = []

        self.stats = defaultdict(list)

        self.cnf = [[0] * len(set(self.data.y))] * len(set(self.data.y))
        self.feature_importances = [-1] * len(self.data.template_order)


    def merge_results_(self, y_true, predictions):
        accuracy = accuracy_score(y_true=y_true, y_pred=predictions)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true = y_true, y_pred = predictions,
            average='macro'
        )

        self.stats['accuracies'].append(accuracy)
        self.stats['precisions'].append(precision)
        self.stats['recalls'].append(recall)
        self.stats['fscores'].append(fscore)

        print('{:.4f} {:.4f} {:.4f} {:.4f}'.format(accuracy, precision, recall, fscore));

        self.cnf = map(add, self.cnf, confusion_matrix(y_true, predictions))


    def merge_importances_(self, importances, idxs):
        for i,v in enumerate(importances):
            #self.feature_importances[idxs[i]] += v
            if (self.feature_importances[idxs[i]] == -1):
                self.feature_importances[idxs[i]] = 0;
            self.feature_importances[idxs[i]] += v
#        self.feature_importances = map(
#            add,
#            zip(importances, self.feature_importances)
#        )
#            self.feature_importances = map(
#                lambda (a,b): a+b \
#                    if a != -1 \
#                    else b, \
#                zip(importances, self.feature_importances)
#            )


    def fit_evaluate_(self, X_train, X_test, y_train, y_test):
        self.clf.fit(X_train, y_train)

        preds = self.clf.predict(X_test)

        self.merge_results_(y_test, preds)
        self.merge_importances_(self.clf.feature_importances_)


    def run(self):
        self.feature_importances = [-1] * len(self.data.template_order)

        oobs = []

        for shuffle_idx in range(self.n_shuffles):
            print('shuffle {} of {}'.format(shuffle_idx+1, self.n_shuffles))

            if self.n_splits > 0:
                kfold = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state
                )

                split_idx = 0
                for train_indices, test_indices in kfold.split(self.data.X, self.data.y):
                    split_idx += 1
                    print('split {} of {}, [{}/{}] '.format(
                        split_idx, self.n_splits,
                        len(train_indices), len(test_indices)
                    ), end='')

                    X_train, X_test, y_train, y_test, template_idxs = \
                            split_data(
                                self.data,
                                train_indices,
                                test_indices,
                                self.reject_templates
                            )



                    self.clf.fit(X_train, y_train)
                    print('OOB predictive accuracy: {} '.format(self.clf.oob_score_), end='')
                    oobs.append(self.clf.oob_score_)
                    #imp1 = copy.deepcopy(self.clf.feature_importances_)
                    preds = self.clf.predict(X_test)
                    #imp2 = copy.deepcopy(self.clf.feature_importances_)
                    #logging.info('MEASURE IMPORTANCES NOW')
                    #Tracer()()

                    self.merge_results_(y_test, preds)
                    self.merge_importances_(self.clf.feature_importances_,
                            template_idxs)

                    #self.fit_evaluate_(X_train, X_test, y_train, y_test)

                    #logging.info('accuracy: {}'.format(
                    #    np.mean(self.accuracies[split_idx]))
                    #)
            print('  {} {} res acc: {:.4f} {:.4f}  precision: {:.4f} {:.4f}  recall: {:.4f} {:.4f}  fscore: {:.4f} {:.4f}'.format(
                np.mean(oobs), np.std(oobs),
                np.mean(self.stats['accuracies'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['accuracies'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['precisions'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['precisions'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['recalls'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['recalls'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['fscores'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['fscores'][shuffle_idx:shuffle_idx+self.n_shuffles]),
            ))

        print ('{} {}'.format(np.mean(oobs),np.std(oobs)))

    def set_classifier(self, clf):
        self.clf = clf


    def print_stats(self):
        for k,v in self.stats.iteritems():
            print('{}: {} std. {}'.format(k, np.mean(v), np.std(v)))

