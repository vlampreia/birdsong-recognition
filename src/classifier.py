from scikits.audiolab import Sndfile
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import datetime
import pylab
import cv2
import os
import sys
from skimage.feature import match_template
from sklearn.preprocessing import normalize

from sklearn.ensemble import RandomForestClassifier

from IPython.core.debugger import Tracer

import sys

#-------------------------------------------------------------------------------

def load_pcm(path):
    wave = Sndfile(path, "r")
    pcm = wave.read_frames(wave.nframes)
    wave.close()
    if wave.channels is not 1:
        pcm = pcm[:,0]
    return (pcm, wave.samplerate)

#-------------------------------------------------------------------------------

def time_ticks(x, pos):
    d = datetime.timedelta(seconds=x)
    return str(d)

#-------------------------------------------------------------------------------

def make_specgram(pcm, samplerate):
    fs = samplerate
    nfft = 512
    window = np.hamming(512)
    noverlap = 512 * 0.75
    vmin = None
    vmax = None

    min_freq = 100
    max_freq = 10000

    pxx, freqs, times = matplotlib.mlab.specgram(
            pcm, NFFT=nfft, Fs=fs, noverlap=noverlap, window=window
            )

    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    pxx = pxx[freq_mask,:]

    pxx = 10*np.log10(pxx)
    pxx = np.array(pxx, dtype=np.uint8)

    freqs = freqs[freq_mask]

    return (pxx, freqs, times)

#-------------------------------------------------------------------------------

def plotSpecgram(pxx, freqs, times, log=False):
    fig, ax = plt.subplots()
    norm = None
    if log:
        norm = LogNorm()

    extent = [times.min(), times.max(), freqs.min(), freqs.max()]
    im = ax.imshow(pxx, extent=extent, origin='lower', aspect='auto',
            cmap=pylab.get_cmap('Greys_r'), norm=norm
            )

    cbar = fig.colorbar(im)
    cbar.set_label('dB')
    ax.axis('tight')

    ax.set_xlabel('time h:mm:ss')
    ax.set_ylabel('kHz')

    scale = 1e3
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)

    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)

    plt.show()

#-------------------------------------------------------------------------------

def plotWavAndSpecgram(pcm, fs):
    fig, ax = plt.subplots(2,1)
    pxx, freqs, times = makeSpecgram(pcm, fs)
    extent = [times.min(), times.max(), freqs.min(), freqs.max()]
    Time = np.linspace(0, len(pcm)/fs, num=len(pcm))
    im0 = ax[0].plot(Time, pcm, 'k')
    im1 = ax[1].imshow(pxx, origin='lower', extent=extent, aspect='auto', cmap=pylab.get_cmap('Greys'), norm=LogNorm())

    ax[0].axis('tight')
    ax[0].set_xlabel('time h:mm:ss')
    ax[0].set_ylabel('dB')
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax[0].xaxis.set_major_formatter(formatter)

    ax[1].axis('tight')
    ax[1].set_xlabel('time h:mm:ss')
    ax[1].set_ylabel('kHz')
    scale=1e3
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax[1].yaxis.set_major_formatter(ticks)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax[1].xaxis.set_major_formatter(formatter)

    plt.show()

#-------------------------------------------------------------------------------

def filterSpecgram(pxx, freqs, times, threshold):
    median_times = np.median(pxx, axis=0)
    median_freqs = np.median(pxx, axis=1)

    fpxx = pxx.copy();
    for i in range(fpxx[:,0].size):
        for j in range(fpxx[0,:].size):
            fpxx[i,j] = (fpxx[i,j] > median_times[j]*threshold and
                         fpxx[i,j] > median_freqs[i]*threshold)
    return fpxx

#-------------------------------------------------------------------------------

def plotRow(graphs, labels=None):
    fig, ax = plt.subplots(1, len(graphs))

    for i in xrange(len(graphs)):
        ax[i].axis('off')
        ax[i].imshow(
            graphs[i], cmap=pylab.get_cmap('Greys')
        );
        if labels is not None: ax[i].set_title(labels[i])

    fig.show()

#-------------------------------------------------------------------------------

def plotMultiple(graphs, window, label=None, labels=None):
    if len(graphs) > 3:
        fig, ax = plt.subplots((len(graphs)+1)/2, 2)
    else:
        fig, ax = plt.subplots(len(graphs), 1)

    if label: fig.suptitle(label)
    for i in xrange(ax.size):
        if len(graphs) > 3:
            _ax = ax[i%(len(graphs)+1)/2][i%2]
        else:
            _ax = ax[i]
        _ax.axis('off')

    for i in xrange(len(graphs)):
        if len(graphs) > 3:
            _ax = ax[i%len(graphs)/2][i%2]
        else:
            _ax = ax[i]

        _ax.imshow(
            graphs[i][window[0][0]:window[0][1],window[1][0]:window[1][1]],
            cmap=pylab.get_cmap('Greys')
        );
        if labels is not None: _ax.set_title(labels[i])

    fig.show()

#-------------------------------------------------------------------------------

def filter_specgram(im):
    t_blockSize = 11
    t_C = 5
    b_kernelsize = 5

    im_blur = cv2.medianBlur(im, b_kernelsize)
    im_thresh = cv2.threshold(im_blur, 255*0.65, 255, cv2.THRESH_BINARY)[1]
    im_thresh = cv2.dilate(im_thresh, np.ones((6,6)))
    im_thresh = cv2.erode(im_thresh, np.ones((6, 6)))
    #im_thresh = cv2.adaptiveThreshold(
    #    im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #    t_blockSize, t_C
    #)

    return im_thresh

#-------------------------------------------------------------------------------

def write_specgram(pxx, fname):
    dpath = os.path.split(fname)[0]
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    try:
        matplotlib.image.imsave(
            fname,
            10*np.log10(pxx),
            origin='lower',
            cmap=pylab.get_cmap('Greys')
            )
    except ValueError as err:
        print '\terror writing specgram file: ', err
    except:
        print '\terror writing specgram file:', sys.exc_info()[0]

#-------------------------------------------------------------------------------

def load_specgram(fname):
    print 'loading specgram ', fname
    im = cv2.imread(fname, 0)
    return im

# todo
# walkdir func, get filenames to process
# create specgram take filename list store
# filter specgram take filename list load store
# ..

#-------------------------------------------------------------------------------

def list_wavs(path):
    paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.wav'):
                paths += [os.path.join(dirpath, filename)]
    return paths

#-------------------------------------------------------------------------------

def create_specgrams(pathlist, specgram_dir, overwrite=False):
    paths = []
    for path in pathlist:
        parentdir = os.path.split(path)
        parentdir = os.path.join(os.path.split(parentdir[0])[1], parentdir[1])
        spath = os.path.splitext(parentdir)[0]
        spath = ''.join([os.path.join(specgram_dir, spath), '.png'])
        paths += [spath]
        if not overwrite and os.path.exists(spath):
            print 'specgram from', path, 'exists as', spath
            continue
        else:
            print 'generating specgram from', path, '->', spath

        pcm, fs = loadPcm(path)
        pxx, freqs, times = makeSpecgram(pcm, fs)
        write_specgram(pxx, spath)
    return paths

#-------------------------------------------------------------------------------

def filter_specgrams(pathlist, specgram_dir, overwrite=False):
    for path in pathlist:
        parentdir = os.path.split(path)
        parentdir = os.path.join(os.path.split(parentdir[0])[1], parentdir[1])
        fpath = os.path.splitext(parentdir)[0]
        fpath = ''.join([os.path.join(specgram_dir, fpath), '_clean.png'])
        if not overwrite and os.path.exists(fpath):
            print 'filtered', path, 'exists as', fpath
            continue
        else:
            print 'filtering', path, '->', fpath

        im = load_specgram(path)
        im_ = filter_specgram(im)
        cv2.imwrite(fpath, im_)

#-------------------------------------------------------------------------------

def process(pcm_dir, specgram_dir):
    pcm_paths = list_wavs(pcm_dir)
    spec_paths = create_specgrams(pcm_paths, specgram_dir, False)
    #filter_specgrams(spec_paths, specgram_dir, True)

    extract_features(spec_paths, 'features')

#    for dirpath, dirnames, filenames in os.walk(pcm_dir):
#        for filename in filenames:
#            if filename.endswith('.wav'):
#                fpath = os.path.join(dirpath, filename)
#                specpath = os.path.join(specgram_dir, os.path.splitext(filename)[0])
#                print 'processing', fpath
#                pcm, fs = loadPcm(fpath)
#                pxx, freqs, times = makeSpecgram(pcm, fs)
#                write_specgram(pxx, specpath)
#                im = load_specgram(''.join([specpath,'.png']))
#                im_ = filter_specgram(im)
#                cv2.imwrite(''.join([specpath,'_clean.png']), im_)

#
# Example usage pattern:
#   >>> pcm, fs = loadPcm('./sample.wav')
#   >>> pxx, freqs, times = makeSpecgram(pxx, freqs, times, True)
#   >>> write_specgram(pxx, './sample.png')
#   >>> im = load_specgram('./sample.png')
#   >>> im_ = filter_specgram(im)

#buildSpectrum("/home/victor/Downloads/woodwren.wav")

def count_samples(samples_dir):
    species_count = {}
    paths = list_wavs(samples_dir)
    for path in paths:
        species = os.path.split(path)[0]
        species = os.path.split(species)[1]

        species_count[species] = species_count.get(species, 0) + 1

    #pprint.pprint(species_count, width=1)
    return species_count

from lxml import html
import requests
import urlparse
import urllib
def scrape_xenocanto(samples_dir, ratings_filter = ['A']):
    url_db = 'http://www.xeno-canto.org'
    page = requests.get(urlparse.urljoin(url_db, '/explore/random'))
    tree = html.fromstring(page.content)

    listings = tree.xpath('//td[@class="sonobox new-species"]')
    for listing in listings:
        rating = listing.xpath('./div[@class="rating"]//li[@class="selected"]/span/text()')
        if len(rating) == 0 or rating[0] not in ratings_filter: continue

        species = listing.xpath('.//span[@class="common-name"]/a/text()')[0]

        url_dl = listing.xpath('.//a[@download]/@href')[0]
        url_dl = urlparse.urljoin(url_db, url_dl)

        filename = listing.xpath('.//a[@download]/@download')[0]
        target_path = os.path.join(samples_dir, species)
        os.mkdir(target_path)
        target_path = os.path.join(target_path, filename)

        print species, '(', rating[0], ')'
        urllib.urlretrieve(url_dl, target_path)
        print '>', target_path
        print ''

#-------------------------------------------------------------------------------

def extract_features(spec_paths, features_dir):
    for path in spec_paths:
        print 'extracting templates from', path
        if not os.path.exists(path):
            print 'ERROR: path doesn\'t exist:', path, 'skipping..'
            continue;

        im = -load_specgram(path)
        features = extract_templates(im)

        filename = os.path.splitext(os.path.split(path)[1])[0]
        fpath = os.path.split(path)[0]
        fpath = os.path.join(fpath, features_dir)
        if not os.path.exists(fpath): os.makedirs(fpath)
        fpath = os.path.join(fpath, filename + '-')
#def extract_templates(spec_paths, features_dir):
#    """Extract templates from specgrams found within a directory and write them
#    to the subdir 'features_dir'.
#    """
#
#    for path in spec_paths:
#        print 'extracting templates from', path
#
#        im = -load_specgram(path)
#        features = extract_templates(im)
#
#        filename = os.path.splitext(os.path.split(path)[1])[0]
#        fpath = os.path.split(path)[0]
#        fpath = os.path.join(fpath, features_dir)
#        if not os.path.exists(fpath): os.makedirs(fpath)
#        fpath = os.path.join(fpath, filename + '-')
#
#        for i in xrange(len(features)):
#            cv2.imwrite(fpath + str(i) + '.png', -features[i])


#    _im = im.copy()
#    contours, hierarchy = cv2.findContours(
#        _im,
#        cv2.RETR_LIST,
#        cv2.CHAIN_APPROX_SIMPLE
#    )
#
#    bounded_contours = map(cv2.boundingRect, contours)
#
#    return bounded_contours

#-------------------------------------------------------------------------------

def draw_featuers(im, contours):
    _im = im.copy()

    for c in contours:
        cv2.rectangle(_im, (c[0],c[1]), (c[0]+c[2],c[1]+c[3]), (255,0,0))

    return _im

#-------------------------------------------------------------------------------

def extract_templates(im):
    tmp = cv2.medianBlur(im, 5)
    tmp = cv2.threshold(tmp, 255*0.65, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(
        tmp,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    templates = []
    #_im = im.copy()
    for i in xrange(len(contours)):
        r = cv2.boundingRect(contours[i])
        if r[2] < 10 or r[3] < 40: continue
        x = im[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        x = cv2.GaussianBlur(x, (0,0), 1.5)
        templates.append(x)
#        cv2.rectangle(_im, (r[0]-10, r[1]-10), (r[0]+r[2]+10, r[1]+r[3]+10), (255,0,0), 1)
#    plt.imshow(_im, aspect='auto')
#    plt.show()

    return templates

#-------------------------------------------------------------------------------
def _writeim(im, name):
    fig = plt.figure(frameon=False)
    sizes = np.shape(im)
    if sizes[0] > sizes[1]:
        w = sizes[0]/sizes[1]
        d = sizes[1]
    else:
        w = sizes[1]/sizes[0]
        d = sizes[0]
    fig.set_size_inches(2, 2, forward=False)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    #ax.set_axis_off()
    #fig.add_axes(ax)
    plt.set_cmap(pylab.get_cmap('Greys'))
    ax.imshow(im)
    print sizes
    plt.savefig(name,dpi=d)


def checkparam(im, thresh_blocksize, thresh_c, blur_kernelsize, blur_g_t, dilate, erode):

    im_cpy = -im.copy()
    #im_cpy = cv2.normalize(im_cpy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_cpy = im_cpy[0:229,0:500]
    
    fig, ax = plt.subplots()
    im = ax.imshow(im_cpy[:,:], extent=[0,500,100,10000], cmap=pylab.get_cmap('Greys'))
    cbar=fig.colorbar(im)
    cbar.set_label('dB')
    ax.axis('tight')
    ax.set_xlabel('time (frames)')
    ax.set_ylabel('kHz')
    scale = 1e3

    ticks=matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)
    #plt.show()
    #plt.savefig('specgram-long.png')

    plot_im = [im_cpy]
    plot_l  = ['im']

    _writeim(im_cpy, 'specgram.png')

    im_blur = cv2.medianBlur(im_cpy, blur_kernelsize)
#    plot_im += [im_blur]
#    plot_l += ['im blur ks:' + str(blur_kernelsize)]

    im_thresh = cv2.adaptiveThreshold(
        -im_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        thresh_blocksize,
        thresh_c
    )
#    plot_im += [im_thresh]
#    plot_l += ['im blur thresh bs: ' + str(thresh_blocksize) + ' c: ' + str(thresh_c)]

    im_thresh_g = cv2.threshold(im_blur, 255*blur_g_t, 255, cv2.THRESH_BINARY)[1]
#   plot_im += [im_thresh_g.copy()]
#    plot_l += ['im blur thresh-g']
    im_contour_src = im_thresh_g.copy()
    #im_thresh_g = cv2.medianBlur(im_thresh_g, blur_kernelsize)
    im_thresh_g = cv2.dilate(im_thresh_g, np.ones((dilate,dilate)))
    im_thresh_g = cv2.erode(im_thresh_g, np.ones((erode, erode)))
#    plot_im += [im_thresh_g]
#    plot_l += ['im blur thresh-g erode t: ' + str(255*blur_g_t) + ' ('+str(blur_g_t)+')']
    _writeim(im_thresh_g, 'specgram-long-clean.png')

    # feature extraction
    tmp = im_contour_src

    plot_im += [tmp]
    plot_l += ['Contours']
    _, contours, _ = cv2.findContours(
        tmp,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    _writeim(tmp, 'contours.png')

    clist = []
    img_cpy2 = im_cpy.copy()
    imgb = cv2.GaussianBlur(im_cpy, (0,0), 2)
    for i in xrange(len(contours)):
        r = cv2.boundingRect(contours[i])
        if r[2] < 10 or r[3] < 20: continue
        cropped = im_cpy[r[1]-10:r[1]+r[3]+10, r[0]-10:r[0]+r[2]+10]
        cv2.rectangle(img_cpy2, (r[0]-10, r[1]-10), (r[0]+r[2]+10, r[1]+r[3]+10), (255,0,0), 1)
        feature = cv2.GaussianBlur(cropped, (0,0), 2)
        clist += [feature]

        tmplres = match_template(imgb, feature)
        tmplres_t = cv2.threshold(tmplres, 0.8, 1, cv2.THRESH_TOZERO)[1]
        #clist += [tmplres_t]
        #ij = np.unravel_index(np.argmax(tmplres), tmplres.shape) #wtf
        #x,y=ij[::-1]
        #cv2.circle(im_blur, (x,y), 5, (255, 0, 0))
    print len(clist)
    #if len(clist) is not 0:
        #plotMultiple(clist,[[0,1000],[0,1000]] )#[[0,1000],[0,1000]])

    plot_im += [img_cpy2]
    plot_l += ['Detected Features']

    _writeim(img_cpy2, 'detected-features.png')

    plot_im += [clist[8]]
    plot_l += ['Selected Feature']

    _writeim(clist[8], 'selected-feature.png')

    tmplres = match_template(imgb, clist[8])
    plot_im += [tmplres]
    plot_l += ['Cross-correlation Map']

    _writeim(tmplres, 'ccm.png')

    tmplres = cv2.threshold(tmplres, 0.7, 1, cv2.THRESH_TOZERO)[1]
    plot_im += [tmplres]
    plot_l += ['Thresholded CCM']

    _writeim(tmplres, 'threshold-ccm.png')

    im_thresh_blur = cv2.medianBlur(im_thresh, blur_kernelsize)
#    plot_im += [im_thresh_blur]
#    plot_l += ['im blur thresh blur ks: ' + str(blur_kernelsize)]

    im_dilate = cv2.dilate(im_thresh, np.ones((dilate, dilate)))
    im_erode = cv2.erode(im_dilate, np.ones((erode, erode)))
#    plot_im += [im_erode]
#    plot_l += ['im blur thresh dilate + erode ks: ' + str(dilate) + ', ' + str(erode)]

#    plotRow(plot_im, plot_l)
    #plotMultiple(plot_im, [[0,1000],[0,1000]],None, plot_l)




# Classification process:
#
#   Pre-processing stage:
#       - Load waveform, get specgram
#       - Clean specgram, find contours and bounds
#       - Extract features, large blocks
#       - Store features to file
#   Training:
#       - Cross-correlate each specgram with each template
#       - gives n-template-dimensional feature vector for each specgram

def cross_correlate(sgram, templates):
    ccms = []

    for template in templates:
        ccm = match_template(sgram, template)
        ccms.append(ccm)

    return ccms;


def get_hdl_from_path(path):
    return os.path.splitext(os.path.split(path)[1])[0]

def get_class_from_path(path):
    return os.path.split(os.path.split(path)[0])[1]

def do(overwrite=False):
    pcm_dir   = './samples'
    sgram_dir = './specgrams'
    X_subdir  = 'features'

    pcm_paths = list_wavs('./samples')

    # load each PCM and construct sgrams
    pcms = {}
    sgrams = {}
    class_templates = {}

    for path in pcm_paths:
        c = get_class_from_path(path)
        hdl = get_hdl_from_path(path)

        print 'class:', c, '-- hdl:', hdl
        print '  load PCM', path
        pcm, fs = load_pcm(path)
        pcms[hdl] = (pcm, fs)

        # TODO: check if specgram for hdl exists on file
        pxx, freqs, times = make_specgram(pcm, fs)
        clean_pxx = filter_specgram(pxx)
        sgrams[hdl] = ((pxx, clean_pxx), (freqs, times))
        print '    made specgram'

        # TODO: check if templates for hdl exist on file
        templates = extract_templates(clean_pxx)
        if c not in class_templates: class_templates[c] = []
        class_templates[c] += templates
        print '    extracted', len(templates), 'templates'
        print ''

    total_templates = sum(len(x) for x in class_templates.values())
    print '\nextracted', total_templates, 'total templates'
    for k,v in class_templates.iteritems():
      n = len(v)
      print '  ', k, ': ', n, '(', (total_templates/n)*100, '%)'


    # now we have all templates for all classes, lets run through each known
    # specgram and construct CCMs
#    print '> cross-correlating templates'
#    class_features = {}
#    for hdl, sgramd in sgrams.iteritems():
#        X_ccm = cross_correlate(sgramd[0], class_templates[hdl])
#        class_features[hdl] = X_ccm
#
#
#    print '> classification test'
#    np_samples = np.array(class_features.values())
#    np_labels = np.array(class_features.keys())
#
#    clf = RandomForestClassifier()
#    Tracer()()
#    # TODO: separate training from test data....
#    clf.fit(np_samples, np_labels)
#    clf.score(np_samples, np_labels)
#    clf.predict(np_sample)
