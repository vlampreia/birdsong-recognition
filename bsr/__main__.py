from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import multiprocessing as mp

import trace
from optparse import OptionParser
import cv2

DIR_SPECTROGRAMS = './spectrograms'
DIR_SAMPLES = './samples'


def cr(i, sgram, template):
    try:
        ccm = cv2.matchTemplate(sgram['sgram'], template['template'], cv2.TM_CCOEFF_NORMED)
        #ccm = match_template(sgram['sgram'], template['template'])
    except Exception as e:
        print 'ERROR matching template {} against {}!!'.format(sgram['hdl'], i)
        print e

    print 'template {} cls: {} from sgram {} on sgram {} class {} max: {}'.format(
            i,
            template['class'],
            template['hdl'],
            sgram['hdl'],
            sgram['class'],
            np.max(ccm)
           )

    #i = i+1
    return (i, np.max(ccm))
    #ccms.append(ccm)
    #ccm_maxs.append(np.max(ccm))

def cross_correlate(sgram, templates):
    ccm_maxs = np.zeros(50)
    #ccms = []
    i = 0


    def cr_collect(result):
        ccm_maxs[result[0]] = result[1]

    #Tracer()()
    pool = mp.Pool(processes=4)
    i = 0;
    for template in templates:
        if i==50: break;
        res = pool.apply_async(cr, (i, sgram, template), callback=cr_collect)
        i = i+1
    pool.close()
    pool.join()


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
    parser.add_option("-s", "--scrape", dest="scrape", action="store_true",
                      help="Scrape random samples from XenoCanto")

    parser.add_option("--stats", dest="stats", action="store_true",
                      help="Print statistics for local samples")

    parser.add_option("-g", "--specgrams", dest="build_sgrams", action="store_true",
                      help="Run spectrogram generation")

    parser.add_option("-c", "--classify", dest="classify", action="store_true",
                      help="Run classifier")

    parser.add_option("-p", "--preprocess", dest="preprocess", action="store_true",
                      help="Run preprocessing steps")

    parser.add_option("-e", "--extract", dest="extract", action="store_true",
                      help="Run feature extraction steps")

    parser.add_option('-t', '--templates', dest='templates', action="store_true",
                      help="Build templates or load")

    (options, args) = parser.parse_args()

    if (options.scrape):
        scraper = XenoCantoScraper()
        scraper.retrieve_random(DIR_SAMPLES)
        print_sample_statistics(DIR_SAMPLES)
        return

    if (options.stats):
        print_sample_statistics(DIR_SAMPLES)
        return

    build_sgrams = False
    if (options.build_sgrams): build_sgrams = True

    build_templates = False
    if (options.templates): build_templates = True

    run_classifier = False
    if (options.classify): run_classifier = True

    run_feature_extraction = False
    if (options.extract): run_feature_extraction = True


    # extracts features and writes to disk
    #process(DIR_SAMPLES, DIR_SPECTROGRAMS)

    overwrite_sgram = False

    pcm_paths = list_wavs(DIR_SAMPLES)

    # load each PCM and construct sgrams
    class_to_idx = {}
    lastidx=1
    class_pcms = {}
    class_sgrams = {}
    class_templates = {}
    all_sgrams = []
    all_templates = {}

    for path in pcm_paths:
        c = get_class_from_path(path)
        hdl = get_hdl_from_path(path)

        if c not in class_to_idx:
            class_to_idx[c] = lastidx
            lastidx = lastidx + 1

        parentdir = os.path.split(path)
        parentdir = os.path.join(os.path.split(parentdir[0])[1], parentdir[1])
        spath = os.path.splitext(parentdir)[0]
        spath = os.path.join(DIR_SPECTROGRAMS, spath)

        print 'class:', c, '-- hdl:', hdl


        path_sgram = ''.join([spath, '.pkl'])
        if build_sgrams and (overwrite_sgram or not os.path.exists(path_sgram)):
            print '  load PCM', path
            pcm, fs = load_pcm(path)
            if c not in class_pcms: class_pcms[c] = []
            class_pcms[c].append((hdl, (pcm, fs)))

            pxx, freqs, times = make_specgram(pcm, fs)
            write_specgram(pxx, freqs, times, spath)
            print '  made specgram'
        else:
            if os.path.exists(path_sgram):
                pxx, freqs, times = load_specgram(spath)

        #TODO: store filtered specgram
        clean_pxx = filter_specgram(pxx)
        #if c not in class_sgrams: class_sgrams[c] = []
        all_sgrams.append({
            'class': c,
            'hdl': hdl,
            'sgram': pxx,
            'clean': clean_pxx,
            'freqs': freqs,
            'times': times
            })
        #class_sgrams[c].append((hdl, ((pxx, clean_pxx), (freqs, times))))


        dir_templates = os.path.join(DIR_SPECTROGRAMS, os.path.join(c, 'features'))
        templates = []
        if build_templates:
            templates = extract_templates(clean_pxx)

            if not os.path.exists(dir_templates): os.makedirs(dir_templates)
            fpath = os.path.join(dir_templates, ''.join([hdl, '-']))

            for i in xrange(len(templates)):
                cv2.imwrite(fpath + str(i) + '.png', -templates[i])

            if len(templates) == 0: continue
            #if c not in class_templates: class_templates[c] = []
            for idx, template in enumerate(templates):
                all_templates[hdl+str(idx)] = {'hdl': hdl, 'class': c, 'template': template}
            print '    extracted {} templates'.format(len(templates))
        else:
            loadc = 0
            if os.path.exists(dir_templates):
                for tmplf in os.listdir(dir_templates):
                    template_hdl = os.path.splitext(tmplf)[0]
                    fpath = os.path.join(dir_templates, tmplf)
                    if os.path.exists(fpath) and template_hdl not in all_templates:
                        template = cv2.imread(fpath, 0)
                        all_templates[template_hdl] = {'hdl': hdl, 'class': c, 'template': template}
                        loadc = loadc + 1
            print '    loaded {} templates'.format(loadc)

        print ''

    total_templates = len(all_templates.keys())
    #total_templates = sum(len(x[1]) for x in class_templates.values())
    print ''
    print 'loaded {} total templates'.format(total_templates)
#    for k,v in class_templates.iteritems():
#      n = len(v[1])
#      print '  {}: {:>4} ({:>3}%)'.format(k, n, (n/total_templates)*100)

    print ''

    print 'total samples: {}, total features: {}'.format(len(all_sgrams), len(all_templates.keys()))

    #X_sgram_templates = np.zeros((len(all_sgrams), len(all_templates)))
    X_sgram_templates = np.zeros((len(all_sgrams), 50))
    labels = np.zeros(len(all_sgrams))
    # now we have all templates for all classes, lets run through each known
    # specgram and construct CCMs
    if (run_feature_extraction):
        print '> cross-correlating templates'
        #class_features = {}
        #Tracer()()
        for idx, sgram in enumerate(all_sgrams):
        #for c in class_sgrams.keys():
            #for sgram in class_sgrams[c]:
                print 'ccm for class {} sgram {}'.format(sgram['class'], sgram['hdl'])
                X_ccm = cross_correlate(sgram, all_templates.values())
                        #[v['template'] for v in all_templates])
                X_sgram_templates[idx] = X_ccm
                labels[idx] = class_to_idx[sgram['class']]
                #class_features[c] = X_ccm

    #TODO: store features
#
#

    if (run_classifier):
        Tracer()()
        """
        feature set: [n_samples, n_features]
        label set: [n_samples]
        """

        print '> classification test'

        clf = RandomForestClassifier()
    #    Tracer()()
    #    # TODO: separate training from test data....
        r1 = clf.fit(X_sgram_templates, labels)
        r2 = clf.score(X_sgram_templates, labels)
        r3 = clf.predict(X_sgram_templates[0])

    Tracer()()

if __name__ == "__main__":
    main()
