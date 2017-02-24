from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

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


class SampleHdl:
    uid   = ''
    label = ''

    def __init__(self, uid, label):
        self.uid = uid
        self.label = labell


class Sample:
    uid = ''
    label = ''
    spectrogram = None

    def __init__(self, uid, label):
        self.uid = uid
        self.label = label

    def get_pcm_path(self, samples_dir):
        fname = ''.join([self.uid, '.wav'])
        return os.path.join(os.path.join(samples_dir, self.label), fname)

    def get_spectrogram_path(self, spectrograms_dir):
        return os.path.join(os.path.join(spectrograms_dir, self.label), self.uid)

    def get_template_dir(self, spectrograms_dir):
        return os.path.join(os.path.join(spectrograms_dir, self.label), 'templates')

    def get_spectrogram(self):
        return spectrogram

    def get_templates(self):
        return self.spectrogram.templates


class Spectrogram:
    src_sample = None
    pxx = None
    freqs = None
    times = None
    templates = None


    def __init__(self, sample, pxx, freqs, times):
        self.src_sample = sample
        self.pxx = pxx
        self.freqs = freqs
        self.times = times
        self.templates = []


class Template:
    idx = -1
    src_spectrogram = None
    im = None

    def __init__(self, src, template, idx):
        self.src_spectrogram = src
        self.im = template
        self.idx = idx


def make_class_mapping(samples):
    idx = 0
    mapping = {}

    for sample in samples:
        if sample.label not in mapping:
            mapping[sample.label] = idx
            idx = idx + 1

    return mapping


def get_sample_from_path(path):
    label = os.path.split(os.path.split(path)[0])[1]
    uid = os.path.splitext(os.path.split(path)[1])[0]
    return Sample(uid, label)


def gather_samples():
    samples  = []
    pcm_paths = list_types(DIR_SAMPLES, ['.wav'])

    for path in pcm_paths:
        samples.append(get_sample_from_path(path))

    return samples


def build_all_spectrograms(samples):
    spectrograms = []

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

        start = time.time()
        pxx, freqs, times = make_specgram(pcm, fs)
        sgram = Spectrogram(sample, pxx, freqs, times)

        sample.spectrogram = sgram
        spectrograms.append(sgram)

    return spectrograms


def load_all_spectrograms(samples, label_filter = None):
    #spectrograms = []

    num_spectrograms = 0

    for sample in samples:
        path = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        if not os.path.exists(''.join([path, '.pkl'])): continue

        if label_filter is not None and sample.label != label_filter:
            continue

        pxx, freqs, times = load_specgram(path)
        sgram = Spectrogram(sample, pxx, freqs, times)

        sample.spectrogram = sgram
        num_spectrograms += 1
        #spectrograms.append(sgram)

    return num_spectrograms
    #return spectrograms


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


#def build_templates(samples, label_filter, all_templates):
#    global g_options
#
#    for sample in samples:
#        if label_filter is not None and sample.label not in label_filter:
#            continue
#
#        if g_options.verbose:
#            print 'Extracting templates for {} {}'.format(sample.uid, sample.label)
#
#        if sample.spectrogram is None:
#            print 'Sample {} has no spectrogram...'.format(sample.uid)
#            continue
#
#        templates = extract_templates(sample.spectrogram.pxx, g_options.templates_interactive)
#        for idx, template in enumerate(templates):
#            t = Template(sample.spectrogram, template, idx)
#            all_templates.append(t)
#            sample.spectrogram.templates.append(t)


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

#    num_proc = 4
#    div = int(math.ceil(len(samples)/num_proc))
#    procs = []
#    proc_templates = []
#
#    for pidx in xrange(num_proc):
#        proc_templates.append([])
#        left = 0 if pidx == 0 else 1 + pidx * div
#        right = (pidx+1) * div
#        print 'proc {}: {} to {}'.format(pidx, left, right)
#        procs.insert(
#            pidx,
#            mp.Process(target=build_templates, args=(samples[left:right], label_filter, proc_templates[pidx]))
#        )
#
#    for pidx in xrange(num_proc):
#        procs[pidx].start()
#
#    for pidx in xrange(num_proc):
#        procs[pidx].join()
#
#    Tracer()()
#
#    return all_templates

    for sample in samples:
        if label_filter is not None and sample.label not in label_filter:
            continue

        if g_options.verbose:
            print 'Extracting templates for {} {}'.format(
                sample.uid, sample.label)

        if sample.spectrogram is None:
            print 'Sample {} has no spectrogram...'.format(sample.uid)
            continue

        #clean = filter_specgram(sample.spectrogram.pxx)
        templates = extract_templates(
            sample.spectrogram.pxx,
            g_options.templates_interactive
        )

        for idx, template in enumerate(templates):
            t = Template(sample.spectrogram, template, idx)
            all_templates.append(t)
            sample.spectrogram.templates.append(t)

    return all_templates


def load_all_templates(samples):
    all_templates = []
    idx = 0

    for sample in samples:
        if sample.spectrogram is None: continue
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): continue

        for f in os.listdir(template_dir):
            if not os.path.splitext(f)[1] == '.png': continue
            if not f.startswith(sample.uid): continue
            #idx = os.path.splitext(f)[0].split('-')
            #if not len(idx) == 2: continue
            im_t = -cv2.imread(os.path.join(template_dir, f), 0)
            t = Template(sample.spectrogram, im_t, idx)
            idx += 1

            sample.spectrogram.templates.append(t)
            #all_templates.extend(t)

    return idx
    #return all_templates


def store_all_templates(samples):
    for sample in samples:
        if sample.spectrogram is None: continue
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): os.makedirs(template_dir)
        for template in sample.spectrogram.templates:
            fname = ''.join([sample.uid, '-', str(template.idx), '.png'])
            path = os.path.join(template_dir, fname)
            cv2.imwrite(path, -template.im)


def print_template_statistics(spectrograms):
    # (total sgrams, total, min/max/avg height, min/max/avg width)
    stats_per_class = {}

    for specgram in spectrograms:
        label = specgram.src_sample.label
        if label not in stats_per_class:
            stats_per_class[label] = [0, 0, -1, -1, 0, -1, -1, 0]

        stats_per_class[label][0] = stats_per_class[label][0] + 1

        for template in specgram.templates:
            stats_per_class[label][1] = stats_per_class[label][1] + 1
            len_x = len(template.im)
            len_y = len(template.im[0])

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

    print '{:_<80}'.format('')

    for i in sorted(stats_per_class.items(), key=operator.itemgetter(1), reverse=True):
        k = i[0]
        v = i[1]
        print fmt_str.format(
                k[:32], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])


#def extract_features(spectrograms, templates, class_mapping):
def extract_features(x, samples, templates, class_mapping):
    """
    Extracts feature vectors for each given sample.
    Computes cross-correlation map for all samples against each template.
    """

    global g_options

    X = np.memmap(x + 'X.memmap', dtype='float32', mode='w+', shape=(len(samples), len(templates)))
    y = np.memmap(x + 'y.memmap', dtype='float32', mode='w+', shape=(len(samples)))
#    X = np.zeros((len(spectrograms), len(templates)))
#    y = np.zeros(len(spectrograms))

    total = len(templates)
    for idx, sgram in enumerate([s.spectrogram for s in samples]):
        print '({}/{}) cross correlating {} {} ({}px) against {} templates'.format(
            idx, len(samples),
            sgram.src_sample.uid, sgram.src_sample.label, len(sgram.times),
            len(templates))
        t1 = time.time()
        X_ccm = cross_correlate(sgram, templates)
        t2 = time.time()
        X[idx] = X_ccm
        y[idx] = class_mapping[sgram.src_sample.label]

        ts = t2-t1
        print '{}m {}s elapsed'.format(ts/60, ts%60)

    return (X, y)


def load_features(x, n_samples, n_templates):
    X = np.memmap(x + 'X.memmap', dtype='float32', mode='r', shape=(n_samples, n_templates))
    y = np.memmap(x + 'y.memmap', dtype='float32', mode='r', shape=(n_samples))
    return X, y


def store_features(X, y):
    print 'store_features NOT IMPLEMENTED'
    pass


def split_and_classify(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    print '> classification test'

    #TODO: RF per class?

    clf = RandomForestClassifier()
    r1 = clf.fit(X_train,  y_train)
    r2 = clf.score(X_test, y_test)

    Tracer()()


def cr(ccm_maxs, sgram, templates):
    divisions = 4
    group_size = int(len(templates)/divisions)
    errors = []

    for i in xrange(divisions):
        left = 0 if i == 0 else i*group_size
        right = (i+1) * group_size - 1 if i < divisions-1 else len(templates)-1

        print '    processing group {}-{}'.format(
            templates[left].idx, templates[right].idx)

        for template in templates[left:right]:
            if len(sgram.pxx) < len(template.im) or \
               len(sgram.pxx[0]) < len(template.im[0]):
                errors.append((template.idx, 'template dim > sgram dim'))
                continue

            ccm = cv2.matchTemplate(
                sgram.pxx,
                template.im,
                cv2.TM_CCOEFF_NORMED
            )
            ccm_maxs[int(template.idx)] = np.max(ccm)

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
            mp.Process(target=cr, args=(ccm_maxs, sgram, templates[left:right]))
        )

    for pidx in xrange(num_proc):
        procs[pidx].start()

    for pidx in xrange(num_proc):
        procs[pidx].join()

    return ccm_maxs;


def select_samples(samples):
    selected_samples = []
    class_counts = defaultdict(int)
    selected_counts = defaultdict(int)

    for sample in samples:
        if sample.spectrogram is not None:
            class_counts[sample.label] += 1
    for sample in samples:
        if class_counts[sample.label] > 10 and sample.spectrogram is not None:
#        if class_counts[sample.label] > 1 and \
#           sample.spectrogram is not None and \
#           selected_counts[sample.label] < 3:
            selected_samples.append(sample)
            selected_counts[sample.label] += 1

    return selected_samples


def main():
    global g_options

    parser = OptionParser()
    parser.add_option("--scrape", dest="scrape", action="store_true",
                      help="Scrape random samples from XenoCanto")
    parser.add_option("--scrape-conserve", dest="scrape2", action="store_true",
                      help="Scrape random samples from XenoCanto, but only labels which already exist")
    parser.add_option("--scrape-period", dest="scrape3", action="store", type="int",
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
                      action="store_true",
                      help="Load features from file")
    parser.add_option("-F", "--make-features", dest="features_build",
                      action="store_true",
                      help="Extract and overwrite features to file. Equivalent to -stF")

    parser.add_option("-c", "--classify", dest="classify", action="store_true",
                      help="Run classifier. Equivalent to -stfc")

    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Print verbose output")
    parser.add_option("-i", "--informative", dest="informative", action="store_true",
                      help="Print informative output")

    parser.add_option("-l", "--filter-label", dest="label_filter", action="store",
                      help="Process only samples of a givel label value")

    parser.add_option("--split", dest="split_specgrams", action="store_true",
                      help="Split spectrograms on 10s boundaries")

    (options, args) = parser.parse_args()
    g_options = options

    if (options.scrape or options.scrape2):
        scraper = XenoCantoScraper()

        def do_scrape():
            if options.scrape:
                scraper.retrieve_random(DIR_SAMPLES)
            elif options.scrape2:
                samples = gather_samples()
                scraper.retrieve_random(DIR_SAMPLES, ['A'], [s.label for s in samples])

        if options.scrape3 is not None:
            while(True):
                do_scrape()
                print 'trying again in {} seconds'.format(options.scrape3)
                time.sleep(options.scrape3)
        else:
            do_scrape()

        print_sample_statistics(DIR_SAMPLES)
        print 'scraped {} samples'.format(scraper.get_pull_count())
        return

    if (options.stats):
        print_sample_statistics(DIR_SAMPLES)

        samples = gather_samples()
        sgrams = load_all_spectrograms(samples)
        templates = load_all_templates(samples)
        print_template_statistics([s.spectrogram for s in samples if s.spectrogram is not None])
        return

    if options.split_specgrams:
        return

    #overwrite_sgram = False

    samples = gather_samples()
    if options.verbose or options.informative:
        print '{} samples'.format(len(samples))

    class_to_idx = make_class_mapping(samples)
    if options.verbose or options.informative:
        print '{} classes'.format(len(class_to_idx))


    all_sgrams = None
    if options.spectrograms_build:
        all_sgrams = build_all_spectrograms(samples)
        if options.verbose or options.informative:
            print 'built {} spectrograms'.format(len(all_sgrams))

        store_all_spectrograms(samples)
    else:
        if options.verbose: print 'loading spectrograms..'
        num_spectrograms = load_all_spectrograms(samples, options.label_filter)
        if options.verbose or options.informative:
            print 'loaded {} spectrograms'.format(num_spectrograms)


    # ignore classes with only one spectrogram
    samples = select_samples(samples)
    if options.verbose:
        print 'filtered down to {} samples'.format(len(samples))

    # TODO: for all loading after templates we need to store the split to file
    samples_train, samples_test = train_test_split(samples, test_size=0.8)

    # make sure all test samples are present in train samples, vice versa
    common_labels = [x for x in set([s.label for s in samples_train]) \
                    if x in set([s.label for s in samples_test])]
    samples_train = [s for s in samples_train if s.label in common_labels]
    samples_test = [s for s in samples_test if s.label in common_labels]


    if options.templates_build:
        Tracer()()
        delete_stored_templates(samples_train)
        all_templates = build_all_templates(samples_train, options.label_filter)
        if options.verbose or options.informative:
            print 'extracted {} templates'.format(len(all_templates))
        if options.verbose:
            print_template_statistics([s.spectrogram for s in samples_train if s.spectrogram is not None])

        store_all_templates(samples)
    elif options.templates_load or options.features_build:
        if options.verbose:
            print 'loading templates..'
        num_templates = load_all_templates(samples_train)
        if options.verbose or options.informative:
            print 'loaded {} templates'.format(num_templates)


    templates_train = []
    for sample in samples_train:
        if sample.spectrogram is not None:
            templates_train.extend(sample.spectrogram.templates)

    print '{} test samples, {} train samples {} templates'.format(
        len(samples_test), len(samples_train), len(templates_train))

    if options.features_build:
        X_train, y_train = extract_features(
            'train', samples_train, templates_train, class_to_idx)

        X_test, y_test = extract_features(
            'test', samples_test, templates_train, class_to_idx)


    #if options.features_build:
    #    X, y = extract_features(all_sgrams, all_templates, class_to_idx)
#   #     if options.verbose or options.informative:
#   #         print 'extracted {} features'.format(len(X))
    #    store_features(X, y)
    elif options.features_load or options.classify:
        X_train, y_train = load_features('train', len(samples_train), len(templates_train))
        X_test, y_test = load_features('test', len(samples_test), len(templates_train))
        #X, y = load_features()
        #X, y = load_features(samples, all_sgrams, all_templates)
    Tracer()()

    if options.classify:
        print class_to_idx
        clf = RandomForestClassifier()
        r1 = clf.fit(X_train, y_train)
        print 'score {}'.format(clf.score(X_test, y_test))

        clf2 = ExtraTreesClassifier(n_estimators=500, max_features=4, min_samples_split=3)
        r2 = clf2.fit(X_train, y_train)
        print 'score2 {}'.format(clf2.score(X_test, y_test))

        Tracer()()
        #split_and_classify(X, y, 0.2)


if __name__ == "__main__":
    main()
