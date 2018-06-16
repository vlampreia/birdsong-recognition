from __future__ import print_function

from bsrdata import Sample, Spectrogram, Template

import logging

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

from ClassifierEvaluator import ClfEval, FeatureData, load_feature_data

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


def build_all_spectrograms(samples):
    num_build = 0

    print('{} sg'.format(len(samples)))
    for sample in samples:
        path = sample.get_pcm_path(DIR_SAMPLES)
        if not os.path.exists(path): continue

        sgpath = sample.get_spectrogram_path()
        if os.path.exists(''.join([sgpath, '.pkl'])):
            print('spectrogram exists: {}'.format(sgpath))
            continue

        try:
            pcm, fs = load_pcm(path)
        except IOError as e:
            print('error loading wav: {}'.format(path))
            print(e)
            continue

        left = 00000
        right = left + 50000
        pxx, freqs, times = make_specgram(pcm[left:right], fs)
        sgram = Spectrogram(sample, pxx, freqs, times)

        fig,ax=plt.subplots()
        fig.set_size_inches(10,2)
        fig.tight_layout()
        time = np.arange(0, len(pcm[left:right])) * (1.0 / fs)
        scale = 1e3
        ticks = matplotlib.ticker.FuncFormatter(
                lambda x, pos: '{0:g}'.format(x/scale))
        def timeTicks(x,pos):
            d = datetime.timedelta(seconds=x)
            return datetime.time(0,0,d.seconds).strftime("%M:%S") + '.' + str(d.microseconds)[:2]
        formatter = matplotlib.ticker.FuncFormatter(timeTicks)
        ax.imshow(pxx, extent=[times.min(),times.max(),freqs.min(),freqs.max()],
                aspect='auto', cmap=pylab.get_cmap('Greys'), origin='lower')
        ax.set_aspect('auto')
        ax.set_ylabel('kHz')

        #ax.plot(time, pcm[15000:50000], 'k')
        #ax.set_xlim(0,35000)
        #ax.set_aspect('auto')
        #ax.axis('tight')
        #ax.set_ylabel('dB')
        #ax.get_yaxis().set_visible(False)

        ax.set_xlabel('time m:s')
        ax.yaxis.set_major_formatter(ticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(0,time[len(time)-1])

        plt.tight_layout()
        plt.show()
        plt.savefig('pcm')

#        plt.plot(time, pcm[15000:50000],'k')
#        plt.tight_layout()
#        plt.xlabel('time h:mm:ss')
#        plt.xlim(0,time[len(time)-1])
#        plt.xaxis.set_major_formatter(formatter)

        #plt.show()

        #Tracer()()
        #exit()
        num_build += 1

        sample.get_spectrogram()
        print(np.min(pxx))
        print(np.max(pxx))
        sample._sample.spectrogram._spectrogram = sgram
        sample._sample.spectrogram._path = './test'
        sample._sample.spectrogram.write_to_file('./test')
        sample._sample.spectrogram.spectrogram = sample._sample.spectrogram.load_from_file(sample,'./test')

    return num_build


def build_all_templates(samples, opt, label_filter = None):
    """
    Build all templates from each spectrogram from each sample.
    Sample must have a spectrogram image.
    Will not process samples which don't correspond to the label fitler.

    Stores templates in sample's spectrogram instance.
    Returns a list of all templates.
    """

    all_templates = []

    for sample in samples:
        if label_filter is not None and sample.get_label() not in label_filter:
            continue

        if opt.verbose:
            print('Extracting templates for {} {}'.format(
                sample.get_uid(), sample.get_label()))

        if sample.get_spectrogram() is None:
            print('Sample {} has no spectrogram...'.format(sample.get_uid()))
            continue

        templates = extract_templates(
            sample.get_spectrogram().get_pxx(),
            opt.templates_interactive
        )

    return all_templates


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



def cross_correlate_work__(ccm_maxs, sgram, offset, templates):
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

    print('{}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
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
    Tracer()()

    for pidx in xrange(num_proc):
        left = 0 if pidx == 0 else pidx*div
        right = (pidx+1)*div - 1 if pidx < num_proc-1 else len(templates) - 1
        print('proc {}: {} to {}'.format(pidx, left, right))
        procs.insert(
            pidx,
            mp.Process(target=cross_correlate_work__, args=(ccm_maxs, sgram, left, templates[left:right]))
        )

    def _do_start(_startc):
        if _startc != 0:
            Tracer()()
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


def plot_cnf_matrix(cnf, y, idx_to_class, plot_now=False, normalize=False, show_values=False):
    _cnf = cnf
    if normalize:
        _cnf = np.array(cnf, copy=True, dtype='float')
        sums = _cnf.sum(axis=1)
        _cnf = cnf / sums[:, np.newaxis]

    plt.figure()
    plt.imshow(_cnf, interpolation='nearest', cmap=plt.cm.Blues)
    #plt.colorbar()

    tm = np.arange(len(set(y)))
    labels = [idx_to_class[x] for x in set(y)]

    if show_values:
        thresh = np.max(_cnf) / 2.0
        for i in xrange(len(_cnf)):
            for j in xrange(len(_cnf[0])):
                if _cnf[i][j] == 0:
                    plt.text(j, i, '-', horizontalalignment='center',
                        color='white' if _cnf[i][j] > thresh else 'black')
                else:
                    plt.text(j, i, '{0:0.2f}'.format(_cnf[i][j]), horizontalalignment='center',
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


def merge_results_d(result_a, result_b, repository):
    if result_a.label_map != result_b.label_map:
        print('error, label mappings dont match, intervention required')
        Tracer()()

    class_to_idx = result_a.label_map
    samples_1 = repository.get_samples_by_uid(result_a.ids)
    samples_2 = repository.get_samples_by_uid(result_b.ids)
    templates_1 = repository.get_templates_by_uid(
        result_a.template_order, samples_1)
    templates_2 = repository.get_templates_by_uid(
        result_b.template_order, samples_2)

    Tracer()()
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


def evaluate_clf_params(params, left, data):
    ce = ClfEval(data, 10, 10, None)
    __i = 0
    for param in params:
        clf = RandomForestClassifier(
            n_estimators = param['n_estimators'],
            max_features = param['max_features'],
            min_samples_split = param['min_samples_split'],
            min_samples_leaf = param['min_samples_leaf'],
            max_depth = param['max_depth'],
#            #warm_start=True,
#            #oob_score=True,
#            n_estimators=500,
#            max_features='sqrt',
#            min_samples_split=3,
#            #bootstrap=True,
            n_jobs=4,
            oob_score=True
#            random_state=1
        )

        print('{} doing params: {}/{} {}'.format(left, __i, len(params), param))
        ce.set_classifier(clf)
        ce.run()

        print('{} params: {}/{} {}'.format(left, __i, len(params), param))
        ce.print_stats()
        print('')
        __i += 1


def plot_feature_importances(fi, path, torder, show=False):
    fi_s = [s/100 for s in fi]
    fig = plt.figure()
    fig.set_size_inches(10,5)
    ax = fig.add_subplot(111)
    ax.plot(fi_s)
    prev_sid = ''
    prev_label = ''
    labels = []
    ticks = []
    agr_labels = []
    agr_ticks = []
    idx = 0
    agr_sum = 0
    agr_importances = []
    v = 0
    for i, uid in enumerate(torder):
        sid = uid.split('-')[0]
        label = idx_to_class[data.y[data.ids.index(sid)]]

        if label != prev_label:
            sid = sid + '\n' + label
            if prev_label != '':
                agr_importances.append(agr_sum)
                agr_sum = 0
                ax.vlines(i,0,np.max(fi_s), 'r', 'dotted')
            ax.text(i, np.max(fi_s)-(0.0004+(v%2)*0.0004), label[:12])
            v += 1
            prev_label = label
            agr_ticks.append(idx)
            agr_labels.append(label)
            idx += 1

        if len(labels) == 0 or sid != prev_sid:
            prev_sid = sid
            labels.append(sid)
            ticks.append(i)

    ax.set_ylabel('Importance (%)')
    ax.set_xlabel('Feature number')
    ax.set_xlim(0,len(fi_s))
    ax.set_ylim(0, np.max(fi_s))
    plt.tight_layout()

    if show: plt.show()
    plt.savefig(path)


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
                      action="store_true",
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

    __loaddata = not options.classify

    if __loaddata:
        repository = SampleRepository(
            spectrograms_dir=DIR_SPECTROGRAMS,
            samples_dir=DIR_SAMPLES
        )

        repository.gather_samples()

        if process_scrape_options(options, repository): exit()
        if process_stats_options(options, repository): exit()

        previous_data = load_feature_data(options.data_load) or None

        logging.info('{} samples'.format(len(repository.samples)))

        class_to_idx = previous_data.label_map \
                if previous_data is not None \
                else make_class_mapping(repository.samples)

        idx_to_class = {v: k for k, v in class_to_idx.iteritems()}
        logging.info('{} classes'.format(len(class_to_idx)))


        # filter ids (keep/reject previous incl. labels)
        previous_ids = previous_data.ids if previous_data is not None else []
        preserve = True;

        if preserve:
            logging.info('keeping ids: {}'.format(previous_ids))
            if len(previous_ids) != 0:
                repository.filter_uids(previous_ids, reject=False)
        else:
            logging.info('rejecting ids: {}'.format(previous_ids))
            repository.filter_uids(previous_ids, reject=True)
            logging.info(
                'remove previous ids: filtered down to {} samples'.format(
                    len(repository.samples))
            )

            previous_labels = set([idx_to_class[y] for y in previous_data.y])
            logging.info('rejecting labels: {}'.format(previous_labels))
            repository.filter_labels(previous_labels, reject=True)
            logging.info('filter labels: filtered down to {} samples'.format(
                len(repository.samples)))

            repository.filter_labels(
                ['Eurasian Blackcap', 'Garden Warbler'], reject=False
            )

            repository.reject_by_class_count(at_least=20)
            logging.info('remove low bound: filtered down to {} samples'.format(
                len(repository.samples)))

            repository.keep_n_of_each_class(20)

        if options.spectrograms_build:
            logging.error('UNSUPPORTED ACTION -- NOT STORING SGRAMS DEV ONLY')
            sgram_count = build_all_spectrograms(repository.samples)
            logging.info('built {} spectrograms'.format(sgram_count))
            #repository.store_all()
            #store_all_spectrograms(samples)


        logging.info('filtered down to {} samples'.format(len(repository.samples)))
        print_template_statistics(repository.samples)

        if options.templates_build:
            logging.error("WILL NOT STORE TEMPLATES .. DEV ONLY")
            #Tracer()()
            #delete_stored_templates(samples)
            all_templates = build_all_templates(
                repository.samples, 
                options,
                options.label_filter
            )
            #store_all_templates(samples)
            logging.info('extracted {} templates'.format(len(all_templates)),
                '',
                'vvv after template build vvv')
            print_template_statistics(samples)
            exit()

        repository.reject_by_template_count_per_class(at_least=2000)
        print_template_statistics(repository.samples)

        X = None
        y = None
        ids = None
        if options.merge_results:
            data_1 = load_feature_data('./merged_data_2.pkl')
            data_2 = load_feature_data('./_new_data_1.pkl')
            pids = data_1.ids + data_2.ids
            Tracer()()
            repository.filter_uids(pids, reject=False)
            data = merge_results_d(data_1, data_2, repository)

            pickle_safe(results, './_merged_results.pkl')
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

            pickle_safe(data, options.features_build)
        elif options.features_load:
            data = previous_data


    if options.classify:
        #with open('./cnfdata', 'r') as f:
        #    ce = pickle.load(f)

        #plot_cnf_matrix(ce.cnf, data.y, idx_to_class, normalize=True, show_values=True)
        #Tracer()()

        data = load_feature_data(options.data_load)
        ce = ClfEval(data, 10, 10, None)
        #clf = ExtraTreesClassifier(
        param_grid = {
            'n_estimators': [300],#[10, 500, 5000, 10000],
            'max_features': [0.33],#, 'sqrt', 'log2'],#, len(previous_data.template_order)*0.8],
            'min_samples_split': [2],#, 10, 100],
            'min_samples_leaf': [1],#, 10, 100],
            'max_depth': [None]#, 5, 10, 100, 200]
        }
        __params_l = list(ParameterGrid(param_grid))
        __i=0

        print('')
        for p in __params_l:
            print('  {}'.format(p))
        print('')

        # Evaluate all parameters. !! Results are dumped to console !!
        t1 = time.time()
        evaluate_clf_params(__params_l, 0, data)
        print ('total time {}'.format(time.time()-t1))


        ##############################################################
        # Evaluate performance with removed insignificant templates ..
        ##############################################################
        # 
        # with open('./cnfdata', 'r') as f:
        #     cnf = pickle.load(f)
        # with open('./fidata', 'r') as f:
        #     fi = pickle.load(f)

        # clf = RandomForestClassifier(
        #     n_estimators = 300,
        #     max_features = 0.33,
        #     oob_score=True,
        #     n_jobs=4
        # )
        # ce.set_classifier(clf)
        # ce.run()

        # ce.print_stats()
        # plot_feature_importances(ce.feature_importances, 'imp1', data.template_order)
        # plot_cnf_matrix(ce.cnf, data.y, idx_to_class, normalize=True, 
        #         show_values=True)
        # plt.show()

        # Tracer()()

        # evaluate performance after removing insignificant templates
        # #fimpc = copy.deepcopy(ce.feature_importances)
        # fimpc = copy.deepcopy(fi)
        # print('templates used in clf: {}'.format(len(fi)));
        # rej = [x[0] for x in zip(data.template_order, fi) if x[1] == 0]
        # keep = [x[0] for x in zip(data.template_order, fi) if x[1] != 0]
        # ce.reject_templates = rej
        # #print('templates used in clf: {}'.format(len(ce.feature_importances)));
        # #rej = [x[0] for x in zip(data.template_order, ce.feature_importances) if x[1] == 0]
        # #keep = [x[0] for x in zip(data.template_order, ce.feature_importances) if x[1] != 0]
        # #ce.reject_templates = rej
        # ce.run()
        # print('templates used in clf now: {}'.format(len(ce.feature_importances)));
        # Tracer()()
        # plot_feature_importances([x for x in ce.feature_importances if x != -1], 'imp2', keep)

        # # get top 5 sgrams for species
        # #
        # # tls = [idx_to_class[data.y[data.ids.index(x.split('-')[0])]] for x in
        # # data.template_order]
        # #
        # # indices = [i for i,x in enumerate(tls) if x == 'Commoon Blackbird']
        # #
        # # top5 = zip(*heapq.nlargest(5, enumerate([fi[i] for i in
        # # indices]),key=operator.itemgetter(1)))[0]
        # #
        # # uids = [data.template_order[indices[i]] for i in top5]


if __name__ == "__main__":
    main()
