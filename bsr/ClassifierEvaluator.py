from __future__ import print_function


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from collections import defaultdict
from operator import add


import numpy as np


class FeatureData:
    X = None
    y = None
    ids = None
    label_map = None
    template_order = None


    def __init__(
        self,
        X=None,
        y=None,
        ids=None,
        label_map=None,
        template_order=None
    ):
        self.X              = X
        self.y              = y
        self.ids            = ids
        self.label_map      = label_map
        self.template_order = template_order


    def from_legacy(self, data):
        self.X              = data['X']
        self.y              = data['y']
        self.ids            = data['ids']
        self.label_map      = data['label_map']
        self.template_order = data['template_order']

        return self

def load_feature_data(path):
    """Does not verify file contents....
    """
    if path is None: return None
    if not os.path.exists(path): return None

    data = None
    with open(path, 'r') as f:
        data = pickle.load(f)
    if type(data) == type({}):
        logging.warning('loaded legacy data format, maybe you should store as new?')
        _data = FeatureData().from_legacy(data)
        data = _data

    return data


class ClfEval:
    clf = None

    cnf = None

    feature_importances = None

    stats = None

    n_shuffles = 0
    n_splits = 0

    data = None

    random_state = None

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

        print('{:.4f} {:.4f} {:.4f} {:.4f}'.format(
            accuracy, precision, recall, fscore)
        )

        self.cnf = map(add, self.cnf, confusion_matrix(y_true, predictions))


    def merge_importances_(self, importances, idxs):
        for i,v in enumerate(importances):
            #self.feature_importances[idxs[i]] += v
            if (self.feature_importances[idxs[i]] == -1):
                self.feature_importances[idxs[i]] = 0
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
                splits = kfold.split(self.data.X, self.data.y)
                for train_indices, test_indices in splits:
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


def split_data(data, train_indices, test_indices, reject_templates):
    X = np.array(data.X)
    y = np.array(data.y)
    ids = np.array(data.ids)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    ids_train = ids[train_indices]
    ids_test  = ids[test_indices]

    used_templates = []
    template_idxs = []
    for idx, uid in enumerate(data.template_order):
        if uid in reject_templates: continue
        if uid.split('-')[0] in ids_train:
            template_idxs.append(idx)
            used_templates.append(uid)

    X_train_t = np.zeros((len(X_train), len(template_idxs)))
    X_test_t =  np.zeros((len(X_test),  len(template_idxs)))

    for i, v in enumerate(X_train):
        X_train_t[i] = v[template_idxs]
    for i, v in enumerate(X_test):
        X_test_t[i] = v[template_idxs]

    return X_train_t, X_test_t, y_train, y_test, template_idxs
