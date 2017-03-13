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
            print 'WARNING did not load sample {}'.format(uid)

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
            print 'WARNING did not load template {}'.format(uid)

    return templates


def build_all_spectrograms(samples):
    num_build = 0

    for sample in samples:
        path = sample.get_pcm_path(DIR_SAMPLES)
        if not os.path.exists(path): continue

        sgpath = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        if os.path.exists(''.join([sgpath, '.pkl'])):
            print 'spectrogram exists: {}'.format(sgpath)
            continue

        try:
            pcm, fs = load_pcm(path)
        except IOError as e:
            print 'error loading wav: {}'.format(path)
            print e
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
            print 'Extracting templates for {} {}'.format(
                sample.get_uid(), sample.get_label())

        if sample.spectrogram is None:
            print 'Sample {} has no spectrogram...'.format(sample.get_uid())
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
    used_labels = set([idx_to_class[l] for l in data['y']])
    templates_per_class = defaultdict(int)
    for tuid in data['template_order']:
        suid = tuid.split('-')[0]
        templates_per_class[data['y'][data['ids'].index(suid)]] += 1

    fmt_str = '{:<32}  {:<5}  {:<5}'
    print fmt_str.format('label', 'sgrms', 'templates')
    print '{:_<82}'.format('')
    for k,v in templates_per_class.iteritems():
        print fmt_str.format(idx_to_class[k], len([l for l in data['y'] if l == k]), v)
    print '{:-<82}'.format('')
    print fmt_str.format('TOTAL', len(data['y']), len(data['template_order']))


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

    print '{:<32} {:<5} {:<5}   {:<5} {:<5} {:<5}   {:<5} {:<5} {:<5}'.format(
        '', '', '', 'y_dim', '', '', 'x_dim', '', ''
    )

    fmt_str = '{:<32} {:<5} {:<5}   {:<5} {:<5} {:<5}   {:<5} {:<5} {:<5}'
    print fmt_str.format(
        'label', 'sgrms', 'count', 'min', 'max', 'avg', 'min', 'max', 'avg'
    )

    print '{:_<82}'.format('')

    for i in sorted(stats_per_class.items(), key=operator.itemgetter(1), reverse=True):
        k = i[0]
        v = i[1]
        print fmt_str.format(
                k[:32], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])


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
        print '({}/{}) cross correlating {} {} ({}px) against {} templates'.format(
            idx+1, len(samples),
            sample.get_uid(), sample.get_label(), len(sgram.times),
            len(templates))
        t1 = time.time()
        X_ccm = cross_correlate(sgram, templates)
        t2 = time.time()
        X[idx] = list(X_ccm)
        y[idx] = class_mapping[sample.get_label()]

        ts = t2-t1
        print 'EF -- elapsed: {} {} m,s'.format(ts/60, ts%60)
        print ''

    ids = [s.get_uid() for s in samples]

    ts = time.time() - t0
    print 'total time {}m {}s'.format(ts/60, ts%60)

    return (X, y, ids)


def store_features(X, y):
    print 'store_features NOT IMPLEMENTED'
    pass


def split_and_classify(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    print '> classification test'

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

        print '    processing group {}-{}'.format(
            offset + left, offset + right)

        for idx, template in enumerate(templates[left:right]):
            if len(sgram.pxx) < len(template.im) or \
               len(sgram.pxx[0]) < len(template.im[0]):
                errors.append((template.idx, 'template dim > sgram dim'))
                continue

            ccm = cv2.matchTemplate(
                sgram.pxx,
                template.im,
                cv2.TM_CCOEFF_NORMED
            )
            ccm_maxs[offset + idx] = np.max(ccm)

    if len(errors) > 0:
        print '    Errors:'
        for e in errors:
            print '    {}'.format(e[0])
            #TODO: write to file or smth


def cross_correlate(sgram, templates):
    ccm_maxs = mp.Array('d', len(templates))

    num_proc = 4
    div = int(len(templates)/num_proc)
    procs = []

    for pidx in xrange(num_proc):
        left = 0 if pidx == 0 else pidx*div
        right = (pidx+1)*div - 1 if pidx < num_proc-1 else len(templates) - 1
        print 'proc {}: {} to {}'.format(pidx, left, right)
        procs.insert(
            pidx,
            mp.Process(target=cr, args=(ccm_maxs, sgram, left, templates[left:right]))
        )

    def _do_start(_startc):
        if _startc > 3:
            try_again = False
            print 'ERROR COULD NOT START PROCESSES {} TIMES.'.format(_startc)
            Tracer()()
            if not try_again: return

        for pidx in xrange(num_proc):
            try:
                procs[pidx].start()
            except OSError as e:
                print 'Error: {}'.format(e)
                if pidx is 0:
                    _do_start(_startc + 1)
                else:
                    print 'WARNING: pidx != 0: {}'.format(pidx)
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


def split_data(data, train_indices, test_indices):
    X = np.array(data['X'])
    y = np.array(data['y'])
    ids = np.array(data['ids'])

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    ids_train = ids[train_indices]
    ids_test  = ids[test_indices]

    used_templates = []
    template_ids = []
    for idx, uid in enumerate(data['template_order']):
        if uid.split('-')[0] in ids_train:
            template_ids.append(idx)
            used_templates.append(uid)

    X_train_t = np.zeros((len(X_train), len(template_ids)))
    X_test_t =  np.zeros((len(X_test),  len(template_ids)))

    for i, v in enumerate(X_train):
        X_train_t[i] = v[template_ids]
    for i, v in enumerate(X_test):
        X_test_t[i] = v[template_ids]

    return X_train_t, X_test_t, y_train, y_test, used_templates


def plot_feature_importances(importances, idx_to_class, data, plot_now=False):
    prev_sid = ''
    prev_label = ''
    labels = []
    ticks = []
    agr_labels = []
    agr_ticks = []
    idx = 0
    agr_sum = 0
    agr_importances = []

    for i, uid in enumerate(data['template_order']):
        sid = uid.split('-')[0]
        label = idx_to_class[data['y'][data['ids'].index(sid)]]

        if label != prev_label:
            sid = sid + '\n' + label
            prev_label = label
            if prev_label != '':
                agr_importances.append(agr_sum)
                agr_sum = 0
            agr_ticks.append(idx)
            agr_labels.append(label)
            idx += 1

        if len(labels) == 0 or sid != prev_sid:
            prev_sid = sid
            labels.append(sid)
            ticks.append(i)

        agr_sum += importances[i]

    Tracer()()
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow([agr_importances]*2, interpolation='none', cmap=plt.cm.Blues)
    ax[0].set_xticks(agr_ticks)
    ax[0].set_xticklabels(agr_labels, rotation=45)
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])

    cax = ax[1].contourf([importances]*2, interpolation='none', cmap=plt.cm.Blues)
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels, rotation=45)
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    fig.colorbar(cax, ax=ax[1])
    fig.tight_layout()

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
    #repository.load_spectrograms()
    #repository.load_templates()
    print_template_statistics(repository.samples)

    return True


def retrieve_previous_data(path):
    if path is None: return None
    data = None

    if not os.path.exists(path):
        print 'Previous data path {} does not exist.'.format(path)
        return None

    with open(path, 'r') as f:
        data = pickle.load(f)

    return data


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

    parser.add_option("--merge", dest="merge_results", action="store_true")


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

    previous_data = retrieve_previous_data(options.features_load) or None

    logging.info('{} samples'.format(len(repository.samples)))

    class_to_idx = previous_data['label_map'] \
            if previous_data is not None \
            else make_class_mapping(repository.samples)

    idx_to_class = {v: k for k, v in class_to_idx.iteritems()}
    logging.info('{} classes'.format(len(class_to_idx)))

    previous_ids = previous_data['ids'] if previous_data is not None else []
    logging.info('rejecting ids: {}'.format(previous_ids))
    repository.filter_labels(previous_ids, reject=True)
    repository.reject_by_class_count(at_least=20)
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

    all_templates = get_first_n_templates_per_class(repository, 3000)

    templates_per_class = defaultdict(list)
    for t in all_templates: templates_per_class[t.get_src_sample().get_label()].append(t)
    for k, v in templates_per_class.iteritems(): print k, len(v)

    Tracer()()


#    for sample in repository.samples:
#        if selected_templates_per_class[sample.get_label()] >= 3000: continue
#        all_templates.extend(sample.get_spectrogram().get_templates()[:2500])
#        #all_templates.extend(sample.spectrogram.templates[:2500])
#        selected_templates_per_class[sample.get_label()] += min(2500,len(sample.spectrogram.templates))


    if len(all_templates) is 0:
        print 'Loaded no raw templates'
    else:
        print 'Keeping {} templates'.format(len(all_templates))
        print ''
        print 'vvv Using Data vvv'
        print_template_statistics(samples)
        print ''

    X = None
    y = None
    ids = None
    if options.merge_results:
        print '{:-<82}'.format('')
        print 'ENTER MERGE MODE'
        print '{:-<82}'.format('')

        filenames = [
            './engineered_data.pkl',
            './merged_data.pkl'
        ]

        with open(filenames[0], 'r') as f:
            data_1 = pickle.load(f)
        with open(filenames[1], 'r') as f:
            data_2 = pickle.load(f)

        print '\nLoad file {}'.format(filenames[0])
        print_loaded_data(data_1, idx_to_class)
        print '\nLoad file {}'.format(filenames[1])
        print_loaded_data(data_2, idx_to_class)

        data_merged = {}

        # cross data_1 samples with data_2 templates
        _samples   = find_samples(data_1['ids'])
        load_all_spectrograms(_samples)
        _templates = find_templates(data_2['template_order'])
        _template_ids_1 = [t.get_uid() for t in _templates]
        X_1, y_1, ids_1 = extract_features(_samples, _templates, class_to_idx)

        # cross data_2 samples with data_1 templates
        _samples = find_samples(data_2['ids'])
        load_all_spectrograms(_samples)
        _templates = find_templates(data_1['template_order'])
        _template_ids_2 = [t.get_uid() for t in _templates]
        X_2, y_2, ids_2 = extract_features(_samples, _templates, class_to_idx)

        X_1_c = copy.deepcopy(data_1['X'])
        for i,x in enumerate(X_1):
            X_1_c[i].extend(x)
        X_2_c = copy.deepcopy(data_2['X'])
        for i,x in enumerate(X_2):
            X_2_c[i].extend(x)

        data_merged = {
            'X': X_1_c + X_2_c,
            'y': y_1 + y_2,
            'ids': ids_1 + ids_2,
            'label_map': class_to_idx,
            'template_order': _template_ids_1 + _template_ids_2
        }

        print '{:-<82}'.format('')
        print 'CHECK DATA -- DO NOT FORGET TO DUMP FILE!!!'
        print '{:-<82}'.format('')
        Tracer()()
    else:
        if options.features_build:
            # build all features: template matching of all templates <-> all sgrams
            X, y, ids = extract_features(samples, all_templates, class_to_idx)
            used_templates = [t.get_uid() for t in all_templates]
            Tracer()()
            with open(options.features_build, 'w') as f:
                pickle.dump({
                    'X': X, 'y': y, 'ids': ids, 'label_map': class_to_idx,
                    'template_order': used_templates
                }, f)

        elif options.features_load:
            filename = options.features_load
            with open(filename, 'r') as f:
                data = pickle.load(f)
                X = copy.deepcopy(data['X'])
                y = copy.deepcopy(data['y'])
                ids = copy.deepcopy(data['ids'])
                used_templates = data['template_order']

                print 'LOADED ENGINEERED DATA FILE {}'.format(filename)
                print ''
                print_loaded_data(data, idx_to_class)


    if options.classify:
        cm = [[0] * len(set(data['y']))] * len(set(data['y']))
        feature_importances = [-1] * len(data['template_order'])
        template_uid_to_idx = {v:i for i,v in enumerate(data['template_order'])}
        n_splits = 10

        for i in range(5):
            print '\niteration {}'.format(i)

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
            #skf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            #results = [None] * n_splits
            results = defaultdict(list)
            results_i=0


            #cm = [[0] * len(set(data['y']))] * len(set(data['y']))

            #clf = RandomForestClassifier(n_jobs=4)
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

            #loo = LeaveOneOut()
            #for train_indices, test_indices in loo.split(data['X']):
            #Tracer()()
            for train_indices, test_indices in skf.split(data['X'], data['y']):
                #Tracer()()
                print '  fold {} train {} test {}...'.format(results_i+1, len(train_indices), len(test_indices))

                t1 = time.time()
                X_train, X_test, y_train, y_test, template_uids = split_data(data, train_indices, test_indices)
                print '    split time {}'.format(time.time() - t1)

                t1 = time.time()
                clf.fit(X_train, y_train)
                #oob_err = 1 - clf.oob_score_
                #print 'oob err: {}'.format(oob_err)
                predictions = clf.predict(X_test)
                accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
                precision, recall, fscore, support = precision_recall_fscore_support(
                        y_true = y_test, y_pred = predictions,
                        average='micro'
                        )
                results['accuracy'].append(accuracy)
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['fscore'].append(fscore)
                #results['support'].append(support)
                results_i += 1
                print '    clf pred time {}'.format(time.time() - t1)

                #Tracer()()
                print '    accuracy: {}'.format(accuracy)
                #print 'precision: {}\nrecall: {}\n fscore: {}\n support: {}'.format(
                #    precision, recall, fscore, support)
                #Tracer()()
                _cm = confusion_matrix(y_test, predictions)
                #breaks with non stratified kfold, since matrix dimensions are
                # no longer equal
                cm = map(add, cm, _cm)

                _feature_importances = clf.feature_importances_
                for idx in xrange(len(template_uids)):
                    i = template_uid_to_idx[template_uids[idx]]
                    if feature_importances[i] == -1:
                        feature_importances[i] = _feature_importances[idx]
                    else:
                        feature_importances[i] += _feature_importances[idx]

            print '\n  {} fold cv results:'.format(n_splits)
            for k,v in results.iteritems():
                print '    {}: {} std: {}'.format(k, np.mean(v), np.std(v))


        plot_feature_importances(feature_importances, idx_to_class, data)
        plot_cnf_matrix(cm, y_test, idx_to_class, normalize=True, show_values=True)
        plt.show()

        #X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        #    X, y, ids, test_size=0.2, random_state=0
        #)

        ## remove template match results from feature vectors where the template
        ## does not belong to a training sample
        #for idx, template in reversed(list(enumerate(all_templates))):
        #    if template.src_sample.uid not in ids_train:
        #        for x in X_train: del x[idx]
        #        for x in X_test: del x[idx]

        #print class_to_idx
        #clf = RandomForestClassifier()
        #r1 = clf.fit(X_train, y_train)
        #print 'score {}'.format(clf.score(X_test, y_test))

        #clf2 = ExtraTreesClassifier(n_estimators=500, max_features=4, min_samples_split=3)
        #r2 = clf2.fit(X_train, y_train)
        #print 'score2 {}'.format(clf2.score(X_test, y_test))

        #precision, recall, fscore, support = score(y_test, [clf2.predict(x) for x in X_test])
        #print 'precision: {}'.format(precision)
        #print 'recall: {}'.format(recall)
        #print 'fscore: {}'.format(fscore)
        #print 'support: {}'.format(support)

        Tracer()()
        #split_and_classify(X, y, 0.2)


if __name__ == "__main__":
    main()
