from classifier import *
from preprocessor import *
from utils import *
from xenocantoscraper import XenoCantoScraper

from sklearn.ensemble import RandomForestClassifier

import trace
from optparse import OptionParser

DIR_SPECTROGRAMS = './spectrograms'
DIR_SAMPLES = './samples'


class sample:
    def __init__(type, specgram, templates):
        self.templates = templates
        self.type = type
        self.specgram = specgram

def cross_correlate(sgram, templates):
    ccms = []
    i = 0
    #Tracer()()
    for template in templates:
        ccm = match_template(sgram['sgram'], template['template'])
        print 'template {} cls: {} from sgram {} on sgram {} class {} max: {}'.format(
                i,
                template['class'],
                template['hdl'],
                sgram['hdl'],
                sgram['class'],
                np.max(ccm)
               )
        i = i+1
        ccms.append(ccm)

    return ccms;


def main():
    parser = OptionParser()
    parser.add_option("-s", "--scrape", dest="scrape", action="store_true",
                      help="Scrape random samples from XenoCanto")
    parser.add_option("--stats", dest="stats", action="store_true",
                      help="Print statistics for local samples")
    parser.add_option("-c", "--classify", dest="classify", action="store_true",
                      help="Run the classifier without processing new data")

    (options, args) = parser.parse_args()

    if (options.scrape):
        scraper = XenoCantoScraper()
        scraper.retrieve_random(DIR_SAMPLES)
        print_sample_statistics(DIR_SAMPLES)
        return

    if (options.stats):
        print_sample_statistics(DIR_SAMPLES)
        return

    skip_preprocess = False
    if (options.classify): skip_preprocess = True


    # extracts features and writes to disk
    #process(DIR_SAMPLES, DIR_SPECTROGRAMS)

    overwrite_sgram = False
    build_templates = True

    pcm_paths = list_wavs(DIR_SAMPLES)

    # load each PCM and construct sgrams
    class_pcms = {}
    class_sgrams = {}
    class_templates = {}
    all_sgrams = []
    all_templates = []

    for path in pcm_paths:
        c = get_class_from_path(path)
        hdl = get_hdl_from_path(path)

        parentdir = os.path.split(path)
        parentdir = os.path.join(os.path.split(parentdir[0])[1], parentdir[1])
        spath = os.path.splitext(parentdir)[0]
        spath = os.path.join(DIR_SPECTROGRAMS, spath)

        print 'class:', c, '-- hdl:', hdl

        if overwrite_sgram or not os.path.exists(''.join([spath, '.pkl'])):
            if skip_preprocess: break

            print '  load PCM', path
            pcm, fs = load_pcm(path)
            if c not in class_pcms: class_pcms[c] = []
            class_pcms[c].append((hdl, (pcm, fs)))

            pxx, freqs, times = make_specgram(pcm, fs)
            write_specgram(pxx, freqs, times, spath)
            print '  made specgram'
        else:
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

        #TODO: store templates
        if build_templates:
            templates = extract_templates(clean_pxx)
            if len(templates) == 0: continue
            #if c not in class_templates: class_templates[c] = []
            for template in templates:
                all_templates.append({'hdl': hdl, 'class': c, 'template': template})

            print '    extracted', len(templates), 'templates'

        print ''

    total_templates = len(all_templates)
    #total_templates = sum(len(x[1]) for x in class_templates.values())
    print '\nextracted', total_templates, 'total templates'
#    for k,v in class_templates.iteritems():
#      n = len(v[1])
#      print '  {}: {:>4} ({:>3}%)'.format(k, n, (n/total_templates)*100)

    print ''

    # now we have all templates for all classes, lets run through each known
    # specgram and construct CCMs
    print '> cross-correlating templates'
    class_features = {}
    #Tracer()()
    for sgram in all_sgrams:
    #for c in class_sgrams.keys():
        #for sgram in class_sgrams[c]:
            print 'ccm for class {} sgram {}'.format(sgram['class'], sgram['hdl'])
            X_ccm = cross_correlate(sgram, all_templates)
                    #[v['template'] for v in all_templates])
            class_features[c] = X_ccm
#
#
    print '> classification test'
    np_samples = np.array(list(class_features.values()))
    np_labels = np.array(class_features.keys())
#
    clf = RandomForestClassifier()
#    Tracer()()
#    # TODO: separate training from test data....
    clf.fit(np_samples, np_labels)
    clf.score(np_samples, np_labels)
    clf.predict(np_sample)

if __name__ == "__main__":
    main()
