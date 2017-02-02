from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import multiprocessing as mp

from sklearn.model_selection import train_test_split
import trace
from optparse import OptionParser
import cv2

DIR_SPECTROGRAMS = './spectrograms'
DIR_SAMPLES = './samples'


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

    for sample in samples:
        if sample.spectrogram is None: continue
        template_dir = sample.get_template_dir(DIR_SPECTROGRAMS)
        if not os.path.exists(template_dir): continue

        for f in os.listdir(template_dir):
            if not os.path.splitext(f)[1] == '.png': continue
            idx = f.split('-')
            if not len(idx) == 2: continue
            im_t = cv2.imread(os.path.join(template_dir, f), 0)
            t = Template(sample.spectrogram, im_t, idx)
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


def extract_features(spectrograms, templates, class_mapping):
    X = np.memmap('X.memmap', dtype='float32', mode='w+', shape=(len(spectrograms), len(templates)))
    y = np.memmap('y.memmap', dtype='float32', mode='w+', shape=(len(spectrograms)))
#    X = np.zeros((len(spectrograms), len(templates)))
#    y = np.zeros(len(spectrograms))

    for idx, sgram in enumerate(spectrograms):
        X_ccm = cross_correlate(sgram, templates)
        X[idx] = X_ccm
        y[idx] = class_mapping[sgram.src_sample.label]

    return (X, y)


def load_features(samples):
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


def cr(i, sgram, template):
    try:
        ccm = cv2.matchTemplate(sgram.pxx, template.im, cv2.TM_CCOEFF_NORMED)
    except Exception as e:
        print 'ERROR matching template {} against {}!!'.format(i, sgram.src_sample.uid)
        print e
        return None

    print 'template {} cls: {} from sgram {} on sgram {} class {} max: {}'.format(
            i,
            template.src_spectrogram.src_sample.label,
            template.src_spectrogram.src_sample.uid,
            sgram.src_sample.uid,
            sgram.src_sample.label,
            np.max(ccm)
           )

    #i = i+1
    return (i, np.max(ccm))
    #ccms.append(ccm)
    #ccm_maxs.append(np.max(ccm))


def cross_correlate(sgram, templates):
    ccm_maxs = np.zeros(len(templates))

    ready_list = []
    def cr_collect(result):
        if result is None: return
        ccm_maxs[result[0]] = result[1]
        ready_list.append(result[0])

    results = {}
#    pool = mp.Pool(processes=4)
    for idx, template in enumerate(templates):
        cr_collect(cr(idx, sgram, template))
#        results[idx] = pool.apply_async(cr, (idx, sgram, template), callback=cr_collect)
#        for ready in ready_list:
#            results[ready].wait()
#            del results[ready]
#        ready_list = []

#    pool.close()
#    pool.join()


#    for template in templates:
#        if i == 50: break
#        try:
#            ccm = match_template(sgram['sgram'], template['template'])
#        except Exception as e:
#            print 'ERROR matching template {} against {}!!'.format(sgram['hdl'], i)
#            print e
#
#        print 'template {} cls: {} from sgram {} on sgram {} class {} max: {}'.format(
#                i,
#                template['class'],
#                template['hdl'],
#                sgram['hdl'],
#                sgram['class'],
#                np.max(ccm)
#               )
#        i = i+1
#        ccms.append(ccm)
#        ccm_maxs.append(np.max(ccm))

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
        print_sample_statistics(DIR_SAMPLES)
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

    X = None
    y = None
    if options.features_build:
        X, y = extract_features(all_sgrams, all_templates, class_to_idx)
        if options.verbose or options.informative:
            print 'extracted {} features'.format(len())
        store_features(X, y)
    elif options.features_load or options.classify:
        X, y = load_features(samples)

    if options.classify:
        split_and_classify(X, y, 0.2)

    return

#    pcm_paths = list_wavs(DIR_SAMPLES)
#
#    # load each PCM and construct sgrams
#    class_to_idx = {}
#    lastidx=1
#    class_pcms = {}
#    all_sgrams = []
#    all_templates = {}
#
#    pcm_paths = list_wavs(DIR_SAMPLES)
#
#    for path in pcm_paths:
#        c = get_class_from_path(path)
#        hdl = get_hdl_from_path(path)
#
#        if c not in class_to_idx:
#            class_to_idx[c] = lastidx
#            lastidx = lastidx + 1
#
#    if (options.spectrograms_build):
#        all_sgrams = build_spectrograms(pcm_paths)
#        store_spectrograms(all_sgrams)
#    elif (options.spectrograms_load):
#        all_sgrams = load_spectrograms()
#
#        parentdir = os.path.split(path)
#        parentdir = os.path.join(os.path.split(parentdir[0])[1], parentdir[1])
#        spath = os.path.splitext(parentdir)[0]
#        spath = os.path.join(DIR_SPECTROGRAMS, spath)
#
#        print 'class:', c, '-- hdl:', hdl
#
#        path_sgram = ''.join([spath, '.pkl'])
#        if build_sgrams and (overwrite_sgram or not os.path.exists(path_sgram)):
#            print '  load PCM', path
#            pcm, fs = load_pcm(path)
#            if c not in class_pcms: class_pcms[c] = []
#            class_pcms[c].append((hdl, (pcm, fs)))
#
#            pxx, freqs, times = make_specgram(pcm, fs)
#            write_specgram(pxx, freqs, times, spath)
#            print '  made specgram'
#        else:
#            if os.path.exists(path_sgram):
#                pxx, freqs, times = load_specgram(spath)
#
#        #TODO: store filtered specgram
#        clean_pxx = filter_specgram(pxx)
#        all_sgrams.append({
#            'class': c,
#            'hdl': hdl,
#            'sgram': pxx,
#            'clean': clean_pxx,
#            'freqs': freqs,
#            'times': times
#            })
#
#
#        dir_templates = os.path.join(DIR_SPECTROGRAMS, os.path.join(c, 'features'))
#        templates = []
#        if build_templates:
#            templates = extract_templates(clean_pxx)
#
#            if not os.path.exists(dir_templates): os.makedirs(dir_templates)
#            fpath = os.path.join(dir_templates, ''.join([hdl, '-']))
#
#            for i in xrange(len(templates)):
#                cv2.imwrite(fpath + str(i) + '.png', -templates[i])
#
#            if len(templates) == 0: continue
#            for idx, template in enumerate(templates):
#                all_templates[hdl+str(idx)] = {'hdl': hdl, 'class': c, 'template': template}
#            print '    extracted {} templates'.format(len(templates))
#        else:
#            loadc = 0
#            if os.path.exists(dir_templates):
#                for tmplf in os.listdir(dir_templates):
#                    template_hdl = os.path.splitext(tmplf)[0]
#                    fpath = os.path.join(dir_templates, tmplf)
#                    if os.path.exists(fpath) and template_hdl not in all_templates:
#                        template = cv2.imread(fpath, 0)
#                        all_templates[template_hdl] = {'hdl': hdl, 'class': c, 'template': template}
#                        loadc = loadc + 1
#            print '    loaded {} templates'.format(loadc)
#        print ''
#
#    print ''
#    print 'loaded {} total templates'.format(len(all_templates.keys()))
#    print ''
#    print 'total samples: {}, total features: {}'.format(len(all_sgrams), len(all_templates.keys()))
#
#    all_sgrams = None
#
#    if (build_sgrams):
#        all_sgrams = build_spectrograms(pcm_paths)
#
#    if (build_templates):
#        pass
#
#    X = None
#    y = None
#
#    if (run_feature_extraction):
#        #TODO: store features
#        X, y = extract_features(all_sgrams, all_templates, class_to_idx)
#
#    if (run_classifier):
#        split_and_classify(X, y, 0.2)
#
#    Tracer()()

if __name__ == "__main__":
    main()
