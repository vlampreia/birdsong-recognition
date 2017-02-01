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

from preprocessor import *
from specgram_utils import *
from utils import *




def time_ticks(x, pos):
    d = datetime.timedelta(seconds=x)
    return str(d)


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


#def filterSpecgram(pxx, freqs, times, threshold):
#    median_times = np.median(pxx, axis=0)
#    median_freqs = np.median(pxx, axis=1)
#
#    fpxx = pxx.copy();
#    for i in range(fpxx[:,0].size):
#        for j in range(fpxx[0,:].size):
#            fpxx[i,j] = (fpxx[i,j] > median_times[j]*threshold and
#                         fpxx[i,j] > median_freqs[i]*threshold)
#    return fpxx


def plotRow(graphs, labels=None):
    fig, ax = plt.subplots(1, len(graphs))

    for i in xrange(len(graphs)):
        ax[i].axis('off')
        ax[i].imshow(
            graphs[i], cmap=pylab.get_cmap('Greys')
        );
        if labels is not None: ax[i].set_title(labels[i])

    fig.show()


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


def draw_featuers(im, contours):
    _im = im.copy()

    for c in contours:
        cv2.rectangle(_im, (c[0],c[1]), (c[0]+c[2],c[1]+c[3]), (255,0,0))

    return _im


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