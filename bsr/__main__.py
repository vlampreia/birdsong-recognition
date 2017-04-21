from __future__ import print_function

from bsrdata import Sample, Spectrogram, Template

import logging

from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from SampleRepository import *
from operator import add

from itertools import islice
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix

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

DIR_SPECTROGRAMS = './spectrograms'
DIR_SAMPLES = './samples'

g_options = None



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
                    imp1 = self.clf.feature_importances_
                    preds = self.clf.predict(X_test)
                    imp2 = self.clf.feature_importances_
                    logger.info('MEASURE IMPORTANCES NOW')
                    Tracer()()

                    self.merge_results_(y_test, preds)
                    self.merge_importances_(self.clf.feature_importances_,
                            template_idxs)

                    #self.fit_evaluate_(X_train, X_test, y_train, y_test)

#                    logging.info('accuracy: {}'.format(
#                        np.mean(self.accuracies[split_idx*]))
#                    )
            print('  res acc: {:.4f} {:.4f}  precision: {:.4f} {:.4f}  recall: {:.4f} {:.4f}  fscore: {:.4f} {:.4f}'.format(
                np.mean(self.stats['accuracies'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['accuracies'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['precisions'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['precisions'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['recalls'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['recalls'][shuffle_idx:shuffle_idx+self.n_shuffles]),
                np.mean(self.stats['fscores'][shuffle_idx:shuffle_idx+self.n_shuffles]), np.std(self.stats['fscores'][shuffle_idx:shuffle_idx+self.n_shuffles])
            ))

    def set_classifier(self, clf):
        self.clf = clf


    def print_stats(self):
        for k,v in self.stats.iteritems():
            print('{}: {} std. {}'.format(k, np.mean(v), np.std(v)))


def make_class_mapping(samples):
    idx = 0
    mapping = {}

    for sample in samples:
        if sample.get_label() not in mapping:
            mapping[sample.get_label()] = idx
            idx = idx + 1

    return mapping


def get_sample_from_path(path):
    label = os.path.split(os.path.split(path)[0])[1]
    uid = os.path.splitext(os.path.split(path)[1])[0]
    return Sample(uid, label)


def gather_samples():
    samples  = []
    pcm_paths = list_types(DIR_SAMPLES, ['.wav'])
    pcm_paths = list_types(DIR_SPECTROGRAMS, ['.png'])

    for path in pcm_paths:
        if 'templates' in path: continue;
        samples.append(get_sample_from_path(path))

    return samples


class FeatureData:
    X = None
    y = None
    ids = None
    label_map = None
    template_order = None


    def __init__(self, X=None, y=None, ids=None, label_map=None, template_order=None):
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


def find_samples(ids):
    samples = [None] * len(ids)
    sample_paths = [p for p in list_types(DIR_SPECTROGRAMS, '.png') if 'templates' not in p]
    sample_paths = {os.path.splitext(os.path.split(k)[1])[0]: v for k, v in zip(sample_paths, sample_paths)}

    for idx, uid in enumerate(ids):
        if uid in sample_paths:
            path = sample_paths[uid]
            label = os.path.split(os.path.split(path)[0])[1]
            samples[idx] = Sample(uid, label)
        else:
            print( 'WARNING did not load sample {}'.format(uid))

    return samples


def find_templates(ids):
    templates = [None] * len(ids)
    template_paths = [p for p in list_types(DIR_SPECTROGRAMS, '.png') if 'templates' in p]
    template_paths = {os.path.splitext(os.path.split(k)[1])[0]: v for k, v in zip(template_paths, template_paths)}

    for idx, uid in enumerate(ids):
        if uid in template_paths:
            path = template_paths[uid]
            sample_uid = uid.split('-')[0]
            template_idx = uid.split('-')[1]

            im_t = -cv2.imread(template_paths[uid], 0)
            t = Template(Sample(sample_uid, None), im_t, int(template_idx))
            templates[idx] = t
        else:
            print('WARNING did not load template {}'.format(uid))

    return templates


def build_all_spectrograms(samples):
    num_build = 0

    for sample in samples:
        path = sample.get_pcm_path(DIR_SAMPLES)
        if not os.path.exists(path): continue

        sgpath = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        if os.path.exists(''.join([sgpath, '.pkl'])):
            print('spectrogram exists: {}'.format(sgpath))
            continue

        try:
            pcm, fs = load_pcm(path)
        except IOError as e:
            print('error loading wav: {}'.format(path))
            print(e)
            continue

        pxx, freqs, times = make_specgram(pcm, fs)
        sgram = Spectrogram(sample, pxx, freqs, times)

        num_build += 1

        sample.spectrogram = sgram

    return num_build


#def load_all_spectrograms(samples, label_filter = None):
#    num_spectrograms = 0
#
#    for sample in samples:
#        path = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
#        if not os.path.exists(''.join([path, '.pkl'])): continue
#
#        if label_filter is not None and sample.get_label() != label_filter:
#            continue
#
#        pxx, freqs, times = load_specgram(path)
#        sgram = Spectrogram(sample, pxx, freqs, times)
#
#        sample.spectrogram = sgram
#        num_spectrograms += 1
#
#    return num_spectrograms


def store_all_spectrograms(samples):
    for sample in samples:
        path = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        sgram = sample.spectrogram
        if sgram is None: continue

        write_specgram(sgram.pxx, sgram.freqs, sgram.times, path)


def delete_stored_templates(samples):
    for sample in samples:
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): continue

        for f in os.listdir(template_dir):
            if not os.path.splitext(f)[1] == '.png': continue
            os.remove(os.path.join(template_dir, f))


def build_all_templates(samples, label_filter = None):
    """
    Build all templates from each spectrogram from each sample.
    Sample must have a spectrogram image.
    Will not process samples which don't correspond to the label fitler.

    Stores templates in sample's spectrogram instance.
    Returns a list of all templates.
    """

    global g_options
    all_templates = []

    for sample in samples:
        if label_filter is not None and sample.get_label() not in label_filter:
            continue

        if g_options.verbose:
            print('Extracting templates for {} {}'.format(
                sample.get_uid(), sample.get_label()))

        if sample.spectrogram is None:
            print('Sample {} has no spectrogram...'.format(sample.get_uid()))
            continue

        #clean = filter_specgram(sample.spectrogram.pxx)
        templates = extract_templates(
            sample.spectrogram.pxx,
            g_options.templates_interactive
        )

        for idx, template in enumerate(templates):
            t = Template(sample, template, idx)
            all_templates.append(t)
            sample.spectrogram.templates.append(t)

    return all_templates


def load_all_templates(samples):
    num_templates = 0

    for sample in samples:
        if sample.spectrogram is None: continue
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): continue

        for f in os.listdir(template_dir):
            if not os.path.splitext(f)[1] == '.png': continue
            if not f.startswith(sample.get_uid()): continue

            im_t = -cv2.imread(os.path.join(template_dir, f), 0)
            uid = os.path.splitext(f)[0]
            idx = uid.split('-')[1]
            t = Template(sample, im_t, int(idx))

            sample.spectrogram.templates.append(t)
            num_templates += 1

    return num_templates


#def store_all_templates(samples):
#    for sample in samples:
#        if sample.spectrogram is None: continue
#
#        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
#        if not os.path.exists(template_dir): os.makedirs(template_dir)
#
#        for template in sample.spectrogram.templates:
#            fname = ''.join([template.get_uid(), '.png'])
#            path = os.path.join(template_dir, fname)
#            cv2.imwrite(path, -template.im)


def print_loaded_data(data, idx_to_class):
    used_labels = set([idx_to_class[l] for l in data.y])
    templates_per_class = defaultdict(int)
    for tuid in data.template_order:
        suid = tuid.split('-')[0]
        templates_per_class[data.y[data.ids.index(suid)]] += 1

    fmt_str = '{:<32}  {:<5}  {:<5}'
    print(fmt_str.format('label', 'sgrms', 'templates'))
    print('{:_<82}'.format(''))
    for k,v in templates_per_class.iteritems():
        print(fmt_str.format(idx_to_class[k], len([l for l in data.y if l == k]), v))
    print('{:-<82}'.format(''))
    print(fmt_str.format('TOTAL', len(data.y), len(data.template_order)))


def print_template_statistics(samples):
    # (total sgrams, total, min/max/avg height, min/max/avg width)
    stats_per_class = {}

    for sample in samples:
        specgram = sample.get_spectrogram()
        label = specgram.get_label()
        if label not in stats_per_class:
            stats_per_class[label] = [0, 0, -1, -1, 0, -1, -1, 0]

        stats_per_class[label][0] = stats_per_class[label][0] + 1

        for template in specgram.get_templates():
            stats_per_class[label][1] = stats_per_class[label][1] + 1
            len_x = len(template.get_im())
            len_y = len(template.get_im()[0])

            if stats_per_class[label][2] == -1 or \
                len_x < stats_per_class[label][2]:
                    stats_per_class[label][2] = len_x

            if stats_per_class[label][3] == -1 or \
                len_x > stats_per_class[label][3]:
                    stats_per_class[label][3] = len_x

            stats_per_class[label][4] = (len_x + stats_per_class[label][1] * stats_per_class[label][4]) / (stats_per_class[label][1] + 1)

            if stats_per_class[label][5] == -1 or \
                len_y < stats_per_class[label][5]:
                    stats_per_class[label][5] = len_y

            if stats_per_class[label][6] == -1 or \
                len_y > stats_per_class[label][6]:
                    stats_per_class[label][6] = len_y

            stats_per_class[label][7] = (len_y + stats_per_class[label][1] * stats_per_class[label][7]) / (stats_per_class[label][1] + 1)

    print('{:<32} {:<5} {:<5}   {:<5} {:<5} {:<5}   {:<5} {:<5} {:<5}'.format(
        '', '', '', 'y_dim', '', '', 'x_dim', '', ''
    ))

    fmt_str = '{:<32} {:<5} {:<5}   {:<5} {:<5} {:<5}   {:<5} {:<5} {:<5}'
    print(fmt_str.format(
        'label', 'sgrms', 'count', 'min', 'max', 'avg', 'min', 'max', 'avg'
    ))

    print('{:_<82}'.format(''))

    for i in sorted(stats_per_class.items(), key=operator.itemgetter(1), reverse=True):
        k = i[0]
        v = i[1]
        print(fmt_str.format(
                k[:32], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]))


def extract_features(samples, templates, class_mapping):
    """
    Extracts feature vectors for each given sample.
    Computes cross-correlation map for all samples against each template.
    """

    X = [[None for x in range(len(templates))] for y in range(len(samples))]
    y = [None for x in range(len(samples))]

    t0 = time.time()
    total = len(templates)
    for idx, sample in enumerate(samples):
        sgram = sample.get_spectrogram()
        print('({}/{}) cross correlating {} {} ({}px) against {} templates'.format(
            idx+1, len(samples),
            sample.get_uid(), sample.get_label(), len(sgram.get_times()),
            len(templates)))
        t1 = time.time()
        X_ccm = cross_correlate(sgram, templates)
        t2 = time.time()
        X[idx] = list(X_ccm)
        y[idx] = class_mapping[sample.get_label()]

        ts = t2-t1
        print('EF -- elapsed: {} {} m,s'.format(ts/60, ts%60))
        print('')

    ids = [s.get_uid() for s in samples]

    ts = time.time() - t0
    print('total time {}m {}s'.format(ts/60, ts%60))

    return (X, y, ids)


def store_features(X, y):
    print('store_features NOT IMPLEMENTED')
    pass


def split_and_classify(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    print('> classification test')

    clf = RandomForestClassifier()
    r1 = clf.fit(X_train,  y_train)
    r2 = clf.score(X_test, y_test)

    Tracer()()


def cr(ccm_maxs, sgram, offset, templates):
    divisions = 4
    group_size = int(len(templates)/divisions)
    errors = []

    for i in xrange(divisions):
        left = 0 if i == 0 else i*group_size
        right = (i+1) * group_size - 1 if i < divisions-1 else len(templates)-1

        print('    processing group {}-{}'.format(
            offset + left, offset + right))

        for idx, template in enumerate(templates[left:right]):
            if len(sgram.get_pxx()) < len(template.get_im()) or \
               len(sgram.get_pxx()[0]) < len(template.get_im()[0]):
                errors.append((template.get_idx(), 'template dim > sgram dim'))
                continue

            ccm = cv2.matchTemplate(
                sgram.get_pxx(),
                template.get_im(),
                cv2.TM_CCOEFF_NORMED
            )
            ccm_maxs[offset + idx] = np.max(ccm)

    if len(errors) > 0:
        print('    Errors:')
        for e in errors:
            print('    {}'.format(e[0]))
            #TODO: write to file or smth


def cross_correlate(sgram, templates):
    ccm_maxs = mp.Array('d', len(templates))

    num_proc = 4
    div = int(len(templates)/num_proc)
    procs = []

    for pidx in xrange(num_proc):
        left = 0 if pidx == 0 else pidx*div
        right = (pidx+1)*div - 1 if pidx < num_proc-1 else len(templates) - 1
        print('proc {}: {} to {}'.format(pidx, left, right))
        procs.insert(
            pidx,
            mp.Process(target=cr, args=(ccm_maxs, sgram, left, templates[left:right]))
        )

    def _do_start(_startc):
        if _startc > 3:
            try_again = False
            print('ERROR COULD NOT START PROCESSES {} TIMES.'.format(_startc))
            Tracer()()
            if not try_again: return

        for pidx in xrange(num_proc):
            try:
                procs[pidx].start()
            except OSError as e:
                print('Error: {}'.format(e))
                if pidx is 0:
                    _do_start(_startc + 1)
                else:
                    print('WARNING: pidx != 0: {}'.format(pidx))
                    Tracer()()

    _do_start(0)

    for pidx in xrange(num_proc):
        procs[pidx].join()

    return ccm_maxs;


def select_samples(samples, exception_list=[]):
    """
    Filters the given samples to classes with at least 20 samples
    """
    selected_samples = []
    class_counts = defaultdict(int)

    for sample in samples:
        class_counts[sample.get_label()] += 1

    for sample in samples:
        if class_counts[sample.get_label()] < 20: continue
        if sample.get_uid() in exception_list: continue
        selected_samples.append(sample)

    return selected_samples


def limit_samples(samples):
    """
    Limits the number of samples to 20 per class
    """
    selected_samples = []
    selected_samples_per_class = defaultdict(int)

    for sample in samples:
        if selected_samples_per_class[sample.get_label()] < 20 and \
           sample.spectrogram is not None:
            selected_samples.append(sample)
            selected_samples_per_class[sample.get_label()] += 1

    return selected_samples


def get_first_n_templates_per_class(repository, n):
    all_templates = []
    num_selected_templates_per_class = defaultdict(int)
    template_iterator_per_class = defaultdict(int)
    it_step = 100

    label_sample_d = repository.get_samples_per_label()

    for label, samples in label_sample_d.iteritems():

        while num_selected_templates_per_class[label] < n:
            idx = template_iterator_per_class[label]

            added = 0

            for sample in samples:
                templates = sample.get_spectrogram().get_templates()[idx:idx+it_step]
                all_templates.extend(templates)
                added += len(templates)

                num_selected_templates_per_class[label] += len(templates)

                if num_selected_templates_per_class[label] >= n: break

            if added == 0: break

            template_iterator_per_class[label] += it_step

    return all_templates


def filter_samples(samples, exclude_labels):
    """
    filters down the sample list to four classes, each with at least 2000
    templates
    """

    selected_samples = []
    template_counts = defaultdict(int)

    for sample in samples:
        if sample.spectrogram is None: continue
        template_counts[sample.get_label()] += len(sample.spectrogram.templates)
    #selected_samples = [s for s in samples if template_counts[s.label] <= 3000 and template_counts[s.label] >= 2000]
    selected_samples = [s for s in samples if template_counts[s.get_label()] >= 2000]

    samples_per_class = {}
    for sample in selected_samples:
        if sample.spectrogram is None: continue;
        if sample.get_label() not in samples_per_class:
            samples_per_class[sample.get_label()] = []
        samples_per_class[sample.get_label()].append(sample)

    selected_samples = []
    num=0
    for k,group in samples_per_class.iteritems():
        if num == 4: break
        if exclude_labels is not None and k in exclude_labels: continue
        num += 1
        selected_samples.extend(group)

    return selected_samples


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


def plot_feature_importances(
    importances,
    template_ids,
    idx_to_class,
    data, plot_now=False
):
    prev_sid = ''
    prev_label = ''
    labels = []
    ticks = []
    agr_labels = []
    agr_ticks = []
    idx = 0
    agr_sum = 0
    agr_importances = []

    #for i, uid in enumerate(data.template_order):
    for i, uid in enumerate(template_ids):
        sid = uid.split('-')[0]
        label = idx_to_class[data.y[data.ids.index(sid)]]

        if label != prev_label:
            sid = sid + '\n' + label
            if prev_label != '':
                agr_importances.append(agr_sum)
                agr_sum = 0
            prev_label = label
            agr_ticks.append(idx)
            agr_labels.append(label)
            idx += 1

        if len(labels) == 0 or sid != prev_sid:
            prev_sid = sid
            labels.append(sid)
            ticks.append(i)

        agr_sum += max(importances[i], 0.0)

    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow([agr_importances]*2, interpolation='none', cmap=plt.cm.Blues)
    ax[0].set_xticks(agr_ticks)
    ax[0].set_xticklabels(agr_labels, rotation=45)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    cax = ax[1].bar(range(0, len(importances)), importances)
    #cax = ax[1].contourf([importances]*2, interpolation='none', cmap=plt.cm.Blues)
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels, rotation=45)
#   ax[1].set_yticks([])
#   ax[1].set_yticklabels([])
    #fig.colorbar(cax, ax=ax[1])
    #fig.tight_layout()

    if plot_now: plt.show()


def plot_cnf_matrix(cnf, y, idx_to_class, plot_now=False, normalize=False, show_values=False):

    _cnf = cnf
    if normalize:
        _cnf = np.array(cnf, copy=True, dtype='float')
        _cnf /= _cnf.max()

    plt.figure()
    plt.imshow(_cnf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    tm = np.arange(len(set(y)))
    labels = [idx_to_class[x] for x in set(y)]

    if show_values:
        thresh = np.max(_cnf) / 2.0
        for i in xrange(len(_cnf)):
            for j in xrange(len(_cnf[0])):
                plt.text(j, i, _cnf[i][j], horizontalalignment='center',
                    color='white' if _cnf[i][j] > thresh else 'black')

    plt.xticks(tm, labels, rotation=90)
    plt.yticks(tm, labels)

    if plot_now: plt.show()


def process_scrape_options(options, repository):
    if not options.scrape: return False

    interval = options.scrape_interval or -1
    labels_filter = None
    if options.scrape_conserve:
        labels_filter = [s.get_uid() for s in repository.get_samples()]

    scraper = XenoCantoScraper(
        interval=interval,
        ratings_filter=['A'],
        labels_filter=labels_filter
    )

    scraper.begin_scrape(DIR_SAMPLES)

    return True


def process_stats_options(options, repository):
    if not options.stats: return False

    print_sample_statistics(DIR_SAMPLES)
    print_template_statistics(repository.samples)

    return True


def load_results(path):
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


def store_results_safe(results, path):
    def _mv_bak(path):
        new_path = path + '.bak'
        if os.path.exists(new_path):
            _mv_bak(new_path)
        os.rename(path, new_path)

    if os.path.exists(path):
        _mv_bak(path)

    with open(path, 'w') as f:
        pickle.dump(results, path)


def merge_results(result_a, result_b, repository):
    if result_a.label_map != result_b.label_map:
        print('error, label mappings dont match, intervention required')
        Tracer()()

    class_to_idx = result_a.label_map
    samples_1 = repository.get_samples_by_uid(result_a.ids)
    samples_2 = repository.get_samples_by_uid(result_b.ids)
    templates_1 = repository.get_templates_by_uid(result_a.template_order, samples_1)
    templates_2 = repository.get_templates_by_uid(result_b.template_order, samples_2)

    X_a, y_a, ids_a = extract_features(samples_1, templates_2, class_to_idx)
    X_b, y_b, ids_b = extract_features(samples_2, templates_1, class_to_idx)

    # take special notice to the flipped use of feature vectors due to
    # template matching results concatentation

    for i,x in enumerate(result_a.X):
        X_a[i].extend(x)
    for i,x in enumerate(X_b):
        result_b.X[i].extend(x)

    data = FeatureData()
    data.X = X_a + result_b.X
    data.y = result_a.y + result_b.y
    data.ids = result_a.ids + result_b.ids
    data.label_map = class_to_idx
    data.template_order = result_a.template_order + result_b.template_order


def merge_results(y_true, predictions, results, cnf):
    accuracy = accuracy_score(y_true=y_true, y_pred=predictions)
    precision, recall, fscore, support = precision_recall_fscore_support(
            y_true = y_true, y_pred = predictions,
            average='weighted'
            )
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['fscore'].append(fscore)
    #results['support'].append(support)

    _cnf = confusion_matrix(y_true, predictions)
    cnf = map(add, cnf, _cnf)


def merge_importances(importances, new_importances):
    map(
        lambda (a,b): a+b \
            if a != -1 \
            else b, \
        zip(importances, new_importances)
    )


def shuffle_split_and_classify(data, n_shuffles, n_splits):
    all_cnf = [[0] * len(set(data.y))] * len(set(data.y))
    all_feature_importances = [-1] * len(data.template_order)
    template_uid_to_idx = {v:i for i,v in enumerate(data.template_order)}
    all_results = defaultdict(list)

    for i in range(n_shuffles):
        print('\niteration {}'.format(i+1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
        results = defaultdict(list)

        clf = ExtraTreesClassifier(
            #warm_start=True,
            #oob_score=True,
            n_estimators=500,
            max_features='sqrt',
            min_samples_split=3,
            #bootstrap=True,
            n_jobs=4,
            random_state=None
        )

        split_n=0
        for train_indices, test_indices in skf.split(data.X, data.y):
            print('  fold {} train {} test {}...'.format(split_n+1, len(train_indices), len(test_indices)))

            t1 = time.time()

            X_train, X_test, y_train, y_test, template_uids = split_data(
                data, train_indices, test_indices)

            print('    split time {}'.format(time.time() - t1))

            t1 = time.time()
            clf.fit(X_train, y_train)

            predictions = clf.predict(X_test)
            feature_importances = clf.feature_importances_

            merge_results(y_test, predictions, results, all_cnf)
            merge_importances(feature_importances, clf.feature_importances_)

            print('    clf pred time {}'.format(time.time() - t1))
            print('    accuracy: {} precision: {} recall: {} fscore: {}'.format(accuracy, precision, recall, fscore, support))

            split_n += 1


        print('\n  {} fold cv results:'.format(n_splits))
        for k,v in results.iteritems():
            print('    {}: {} std: {}'.format(k, np.mean(v), np.std(v)))
            all_results[k].extend(v)


    print('\n  {} shuffle iteration results:'.format(n_splits))
    for k,v in all_results.iteritems():
        print('    {}: {} std: {}'.format(k, np.mean(v), np.std(v)))

    plot_feature_importances(all_feature_importances, idx_to_class, data)
    plot_cnf_matrix(cm, y_test, idx_to_class, normalize=True, show_values=True)
    plt.show()


def main():
    global g_options

    parser = OptionParser()
    parser.add_option("--scrape", dest="scrape", action="store_true",
                      help="Scrape random samples from XenoCanto")
    parser.add_option("--scrape-conserve", dest="scrape_conserve", action="store_true",
                      help="Scrape random samples from XenoCanto, but only labels which already exist")
    parser.add_option("--scrape-period", dest="scrape_interval", action="store", type="int",
                      help="Scrape random samples from XenoCanto, at given interval in seconds")

    parser.add_option("--stats", dest="stats", action="store_true",
                      help="Print statistics for local samples")

    parser.add_option("-s", "--load-spectrograms", dest="spectrograms_load",
                      action="store_true",
                      help="Load spectrograms from file")
    parser.add_option("-S", "--make-spectrograms", dest="spectrograms_build",
                      action="store_true",
                      help="Make and overwrite spectrograms to file.")

    parser.add_option("-t", "--load-templates", dest="templates_load",
                      action="store_true",
                      help="Load templates from file")
    parser.add_option("-T", "--make-templates", dest="templates_build",
                      action="store_true",
                      help="Make and overwrite templates to file. Equivalent to -sT")
    parser.add_option("--show-templates", dest="templates_interactive", action="store_true",
                      help="Show template extraction results")

    parser.add_option("-d", dest="data_load",
                      action="store")

    parser.add_option("-f", "--load-features", dest="features_load",
                      action="store",
                      help="Load features from file")
    parser.add_option("-F", "--make-features", dest="features_build",
                      action="store",
                      help="Extract and overwrite features to file. Equivalent to -stF")

    parser.add_option("-c", "--classify", dest="classify", action="store_true",
                      help="Run classifier. Equivalent to -stfc")

    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Print verbose output")
    parser.add_option("-i", "--informative", dest="informative", action="store_true",
                      help="Print informative output")

    parser.add_option("-l", "--filter-label", dest="label_filter", action="store",
                      help="Process only samples of a givel label value")

    parser.add_option("--merge", dest="merge_results", action="store")


    (options, args) = parser.parse_args()
    g_options = options

    logging.basicConfig(level=logging.INFO)

    repository = SampleRepository(
        spectrograms_dir=DIR_SPECTROGRAMS,
        samples_dir=DIR_SAMPLES
    )

    repository.gather_samples()

    if process_scrape_options(options, repository): exit()
    if process_stats_options(options, repository): exit()

    previous_data = load_results(options.data_load) or None

    logging.info('{} samples'.format(len(repository.samples)))

    class_to_idx = previous_data.label_map \
            if previous_data is not None \
            else make_class_mapping(repository.samples)

    idx_to_class = {v: k for k, v in class_to_idx.iteritems()}
    logging.info('{} classes'.format(len(class_to_idx)))

    previous_ids = previous_data.ids if previous_data is not None else []
    logging.info('rejecting ids: {}'.format(previous_ids))
    repository.filter_uids(previous_ids, reject=True)
    logging.info('remove previous ids: filtered down to {} samples'.format(len(repository.samples)))

    previous_labels = set([idx_to_class[y] for y in previous_data.y])
    logging.info('rejecting labels: {}'.format(previous_labels))
    repository.filter_labels(previous_labels, reject=True)
    logging.info('filter labels: filtered down to {} samples'.format(len(repository.samples)))

    repository.reject_by_class_count(at_least=20)
    logging.info('remove low bound: filtered down to {} samples'.format(len(repository.samples)))
    #samples = select_samples(samples, exception_list=previous_ids)

    if options.spectrograms_build:
        logging.error('UNSUPPORTED ACTION')
        exit()
        sgram_count = build_all_spectrograms(samples)
        logging.info('built {} spectrograms'.format(sgram_count))
        store_all_spectrograms(samples)
#    elif options.spectrograms_load:
#        if options.verbose: print 'loading spectrograms..'
#        num_spectrograms = load_all_spectrograms(samples, options.label_filter)
#        logging.info('loaded {} spectrograms'.format(num_spectrograms))


#    if options.spectrograms_load or options.spectrograms_build:
#        print_template_statistics(samples)
#        samples = limit_samples(samples)
#        print_template_statistics(samples)
    repository.keep_n_of_each_class(20)

    # ignore classes with only one spectrogram
    logging.info('filtered down to {} samples'.format(len(repository.samples)))
    print_template_statistics(repository.samples)

    if options.templates_build:
        logging.error('UNSUPPORTED ACTION')
        exit()
        Tracer()()
        delete_stored_templates(samples)
        all_templates = build_all_templates(samples, options.label_filter)
        logging.info('extracted {} templates'.format(len(all_templates)),
            '',
            'vvv after template build vvv')
        print_template_statistics(samples)

        store_all_templates(samples)
#    elif options.templates_load or options.features_build:
#        if options.verbose:
#            print 'loading templates..'
#        num_templates = load_all_templates(samples)
#        if options.verbose or options.informative:
#            print 'loaded {} templates'.format(num_templates)


    #samples = filter_samples(samples, ['Common Whitethroat', 'White-breasted Wood Wren', 'Pale-breasted Spinetail', 'Chestnut-breasted Wren', 'Corn Bunting', 'Ortolan Bunting', 'Common Chiffchaff'])
    #samples = filter_samples(samples, None)
    repository.reject_by_template_count_per_class(at_least=2000)
    print_template_statistics(repository.samples)


#    templates_per_class = defaultdict(list)
#    for t in all_templates: templates_per_class[t.get_src_sample().get_label()].append(t)
#    for k, v in templates_per_class.iteritems(): print k, len(v)


#    for sample in repository.samples:
#        if selected_templates_per_class[sample.get_label()] >= 3000: continue
#        all_templates.extend(sample.get_spectrogram().get_templates()[:2500])
#        #all_templates.extend(sample.spectrogram.templates[:2500])
#        selected_templates_per_class[sample.get_label()] += min(2500,len(sample.spectrogram.templates))



    X = None
    y = None
    ids = None
    if options.merge_results:
        data_1 = load_results('./engineered_data.pkl')
        data_2 = load_results('./merged_data.pkl')
        data = merge_results(data_1, data_2, repository)

        store_results_safe(results, './_merged_results.pkl')
        Tracer()()

    elif options.features_build:
        repository.filter_labels([
            'Eurasian Reed Warbler',
            'Garden Warbler',
            'Western Meadowlark',
            'Eurasian Blackcap'
        ], reject=False)
        all_templates = get_first_n_templates_per_class(repository, 3000)
        logging.info('{} template(s) loaded'.format(len(all_templates)))
        X, y, ids = extract_features(
                repository.samples, all_templates, class_to_idx)
        used_templates = [t.get_uid() for t in all_templates]

        data = {
            'X': X,
            'y': y,
            'ids': ids,
            'label_map': class_to_idx,
            'template_order': used_templates
        }

        store_results_safe(data, options.features_build)
    elif options.features_load:
        data = previous_data


    if options.classify:
        ce = ClfEval(data, 1, 10, 1)
        clf = ExtraTreesClassifier(
        #clf = RandomForestClassifier(
            #warm_start=True,
            #oob_score=True,
            n_estimators=500,
            max_features='sqrt',
            min_samples_split=3,
            #bootstrap=True,
            n_jobs=4,
            random_state=1
        )
        ce.set_classifier(clf)
        ce.run()

        ce.print_stats()

        Tracer()()
        plot_feature_importances(
            ce.feature_importances,
            data.template_order,
            idx_to_class,
            data
        )
        plot_cnf_matrix(ce.cnf, data.y, idx_to_class, normalize=True, show_values=True)
        plt.show()

        #fimpc = copy.deepcopy(ce.feature_importances)
        #print('templates used in clf: {}'.format(len(ce.feature_importances)));
        #rej = [x[0] for x in zip(data.template_order, ce.feature_importances) if x[1] == 0]
        #keep = [x[0] for x in zip(data.template_order, ce.feature_importances) if x[1] != 0]
        #ce.reject_templates = rej
        #ce.run()

        print('templates used in clf now: {}'.format(len(ce.feature_importances)));

        #plot_feature_importances([x for x in ce.feature_importances if x != -1], keep, idx_to_class, data)

        Tracer()()

        #plt.bar(range(0, [x for x in ce.feature_importances if x != -1]), [x for x in ce.feature_importances if x != -1])
        #plt.show()


if __name__ == "__main__":
    main()
