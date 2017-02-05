from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import math
import multiprocessing as mp

from sklearn.model_selection import train_test_split
import trace
from optparse import OptionParser
import cv2
import random

DIR_SPECTROGRAMS = './spectrograms'
DIR_SAMPLES = './samples'

#TODO: fix sample/template distribution


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

        pcm, fs = load_pcm(path)
        pxx, freqs, times = make_specgram(pcm, fs)
        sgram = Spectrogram(sample, pxx, freqs, times)

        sample.spectrogram = sgram
        spectrograms.append(sgram)

    return spectrograms


def load_all_spectrograms(samples):
    spectrograms = []

    for sample in samples:
        path = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        if not os.path.exists(''.join([path, '.pkl'])): continue

        pxx, freqs, times = load_specgram(path)
        sgram = Spectrogram(sample, pxx, freqs, times)

        sample.spectrogram = sgram
        spectrograms.append(sgram)

    return spectrograms


def store_all_spectrograms(samples):
    for sample in samples:
        path = sample.get_spectrogram_path(DIR_SPECTROGRAMS)
        sgram = sample.spectrogram
        write_specgram(sgram.pxx, sgram.freqs, sgram.times, path)


def delete_stored_templates(samples):
    for sample in samples:
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): continue

        for f in os.listdir(template_dir):
            if not os.path.splitext(f)[1] == '.png': continue
            os.remove(os.path.join(template_dir, f))


def build_all_templates(samples):
    all_templates = []

    for sample in samples:
        if sample.spectrogram is None: continue

        #clean = filter_specgram(sample.spectrogram.pxx)
        templates = extract_templates(sample.spectrogram.pxx)
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
            #idx = os.path.splitext(f)[0].split('-')
            #if not len(idx) == 2: continue
            im_t = cv2.imread(os.path.join(template_dir, f), 0)
            t = Template(sample.spectrogram, im_t, idx)
            idx = idx + 1

            sample.spectrogram.templates.append(t)
            all_templates.append(t)

    return all_templates


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


def extract_features(spectrograms, templates, class_mapping):
    X = np.memmap('X.memmap', dtype='float32', mode='w+', shape=(len(spectrograms), len(templates)))
    y = np.memmap('y.memmap', dtype='float32', mode='w+', shape=(len(spectrograms)))
#    X = np.zeros((len(spectrograms), len(templates)))
#    y = np.zeros(len(spectrograms))

    total = len(templates)
    for idx, sgram in enumerate(spectrograms):
        print '({}/{}) cross correlating {} {} against {} templates'.format(
            idx, total,
            sgram.src_sample.uid, sgram.src_sample.label, len(templates))
        X_ccm = cross_correlate(sgram, templates)
        X[idx] = X_ccm
        y[idx] = class_mapping[sgram.src_sample.label]

    return (X, y)


def load_features(samples, spectrograms, templates):
    X = np.memmap('X.memmap', dtype='float32', mode='r', shape=(len(spectrograms), len(templates)))
    y = np.memmap('y.memmap', dtype='float32', mode='r', shape=(len(spectrograms)))
    return X, y
    print 'load_features NOT IMPLEMENTED'
    return None, None


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
    group_size = int(math.ceil(len(templates)/divisions))
    errors = []

    for i in xrange(divisions):
        left = 0 if i == 0 else 1 + i*group_size
        right = (i+1) * group_size

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
    div = int(math.ceil(len(templates)/num_proc))
    procs = []

    for pidx in xrange(num_proc):
        left = 0 if pidx == 0 else 1 + pidx*div
        right = (pidx+1)*div
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


def main():
    parser = OptionParser()
    parser.add_option("--scrape", dest="scrape", action="store_true",
                      help="Scrape random samples from XenoCanto")

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

    (options, args) = parser.parse_args()

    if (options.scrape):
        scraper = XenoCantoScraper()
        scraper.retrieve_random(DIR_SAMPLES)
        print_sample_statistics(DIR_SAMPLES)
        return

    if (options.stats):
        #print_sample_statistics(DIR_SAMPLES)

        samples = gather_samples()
        sgrams = load_all_spectrograms(samples)
        templates = load_all_templates(samples)
        print_template_statistics(sgrams)
        return

    #overwrite_sgram = False

    samples = gather_samples()
    class_to_idx = make_class_mapping(samples)
    if options.verbose or options.informative:
        print '{} samples'.format(len(samples))
        print '{} classes'.format(len(class_to_idx))

    all_sgrams = None
    if options.spectrograms_build:
        all_sgrams = build_all_spectrograms(samples)
        if options.verbose or options.informative:
            print 'built {} spectrograms'.format(len(all_sgrams))

        store_all_spectrograms(samples)
    else:
    #elif options.spectrograms_load or options.templates_build or options.features_build:
        all_sgrams = load_all_spectrograms(samples)
        if options.verbose or options.informative:
            print 'loaded {} spectrograms'.format(len(all_sgrams))

    all_templates = None
    if options.templates_build:
        delete_stored_templates(samples)
        all_templates = build_all_templates(samples)
        if options.verbose or options.informative:
            print 'extracted {} templates'.format(len(all_templates))

        store_all_templates(samples)
    elif options.templates_load or options.features_build:
        all_templates = load_all_templates(samples)
        if options.verbose or options.informative:
            print 'loaded {} templates'.format(len(all_templates))

    # shuffle templates...
    #random.shuffle(all_templates)

    X = None
    y = None
    if options.features_build:
        X, y = extract_features(all_sgrams, all_templates, class_to_idx)
#        if options.verbose or options.informative:
#            print 'extracted {} features'.format(len(X))
        store_features(X, y)
    elif options.features_load or options.classify:
        X, y = load_features(samples, all_sgrams, all_templates)

    if options.classify:
        print class_to_idx
        split_and_classify(X, y, 0.2)


if __name__ == "__main__":
    main()
