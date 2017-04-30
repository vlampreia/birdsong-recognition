import utils
import specgram_utils

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.core.debugger import Tracer

import pylab
def plotMultiple(graphs, label=None, labels=None):
#    if len(graphs) > 3:
#        fig, ax = plt.subplots((len(graphs)+1)/2, 2)
#    else:
        #fig, ax = plt.subplots(len(graphs), 1, sharex=True)
    fig = plt.figure()
    ax = []
    for i, graph in enumerate(graphs):
        _ax = None
        if i > 0: _ax = plt.subplot2grid((2,1), (i,0), sharex = ax[i-1],
                                                       sharey = ax[i-1])
        else: _ax = plt.subplot2grid((2,1), (i,0))
        _ax.axis('off')
        _ax.imshow(graph, cmap=pylab.get_cmap('Greys'), vmin=np.min(graph), vmax=np.max(graph))#, origin='lower')
        _ax.set_aspect('auto')
        if labels is not None: _ax.set_title(labels[i])
        ax.append(_ax)

    fig.tight_layout()
    if label: fig.suptitle(label)
    #for i in xrange(ax.size):
    #    if len(graphs) > 3:
    #        _ax = ax[i%(len(graphs)+1)/2][i%2]
    #    else:
    #        _ax = ax[i]
    #    _ax.axis('off')

    #for i in xrange(len(graphs)):
    #    if len(graphs) > 3:
    #        _ax = ax[i%len(graphs)/2][i%2]
    #    else:
    #        _ax = ax[i]

    #    _ax.imshow(
    #        graphs[i],
    #        cmap=pylab.get_cmap('Greys_r')
    #    );

    #    if labels is not None: _ax.set_title(labels[i])

    plt.show()


def extract_templates(im, interactive = False):
    """
    Extract all templates from a given spectrogram image
    """

    im = np.flipud(im)
#    tmp = cv2.medianBlur(im, 5)
#    tmp = cv2.threshold(tmp, 255*0.65, 255, cv2.THRESH_BINARY)[1]

    im_filtered = filter_specgram(im, interactive)
    _, contours, _ = cv2.findContours(
        im_filtered,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    templates = []

    im_dbg_template_rejected = None
    im_dbg_template_overlay = None
    if interactive:
        im_dbg_template_rejected = im.copy()
        im_dbg_template_overlay = im.copy()

    #im_dbg_template_overlay *= 255/im_dbg_template_overlay.max()


    # apply trunc threshold
    # apply gaussian blur
    # apply binary threshold
    # remove small blobs
    # remove huge blobs
    # for each blob, check surrounding blobs within given radius and add 
    #   (how to choose which to add? what radius?
    smallest = -1
    average_val = np.average(im)
    print 'average: {}'.format(average_val)

    for i in xrange(len(contours)):
        r = cv2.boundingRect(contours[i])

        left = max(0, r[0] - 10)
        top = max(0, r[1] - 10)
        right = min(len(im[0]), r[0] + r[2] + 10)
        bottom = min(len(im), r[1] + r[3] + 10)

        area = r[2] * r[3]

        #TODO: use average values from sgram?
        if area < 50 or area > 10000: # : continue
        #if area > 10000:
            if not interactive: continue
#            cv2.putText(im_dbg_template_rejected, '{}'.format(area),
#                    (left, top), cv2.FONT_HERSHEY_PLAIN, 1.0,
#                    int(np.max(im_dbg_template_rejected)))
            cv2.rectangle(im_dbg_template_rejected, (left,top), (right,bottom), int(np.max(im_dbg_template_rejected)), 1)
            continue

        if smallest == -1 or area < smallest: smallest = area

        x = im[top:bottom, left:right]
        #x = im[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        if np.min(x) >= average_val:
            if not interactive: continue
            cv2.putText(im_dbg_template_rejected, 'v:{}'.format(np.average(x)), (left, top), cv2.FONT_HERSHEY_PLAIN, 1.0, int(np.max(im_dbg_template_rejected)))
            cv2.rectangle(im_dbg_template_rejected, (left,top), (right,bottom), int(np.max(im_dbg_template_rejected)), 1)
            continue
        x = cv2.GaussianBlur(x, (0,0), 1.5)
        templates.append(x)

        if interactive:
            cv2.rectangle(im_dbg_template_overlay, (left, top), (right, bottom), int(np.max(im_dbg_template_overlay)), 1)
        #cv2.rectangle(im_dbg_template_overlay, (r[0]-10, r[1]-10), (r[0]+r[2]+10, r[1]+r[3]+10), (255,0,0), 1)
    if interactive:
        plotMultiple([im_dbg_template_overlay, im_dbg_template_rejected],
        #plotMultiple([im_filtered, im_dbg_template_rejected],
                     None,
                     ['templates', 'rejected'])


#        cv2.namedWindow('orig')
#        cv2.imshow('orig', im_dbg_template_overlay)
#        cv2.namedWindow('rejected')
#        cv2.imshow('rejected', im_dbg_template_rejected)
    #    plt.imshow(im_dbg_template_overlay, aspect='auto')
    #    plt.show()
        print 'smallest: {}'.format(smallest)
    plt_(im_dbg_template_rejected,'reject')
    plt_(im_dbg_template_overlay,'accept')
#        while cv2.waitKey(0) != ord('n'):
#            pass

    return templates


def plt_(im,f):
    im=im[:,:350]
    fig,ax=plt.subplots()
    fig.set_size_inches(4,2)
    fig.tight_layout()
    ax.imshow(im,aspect='auto',cmap=pylab.get_cmap('Greys'))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f)


def filter_specgram(im, interactive = False):
    t_blockSize = 11
    t_C = 5
    b_kernelsize = 5

#    im_blur = cv2.GaussianBlur(im, (b_kernelsize, b_kernelsize), 0)
#    #im_blur = cv2.medianBlur(im, b_kernelsize)
#    tv = 255*0.65
#    im_thresh = cv2.threshold(im_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#    #im_thresh = cv2.threshold(im_blur, tv, 255, cv2.THRESH_BINARY)[1]
#
##    im_thresh = cv2.erode(im_thresh, np.ones((3,3)))
##    im_thresh = cv2.dilate(im_thresh, np.ones((3,3)))
#    im_thresh = cv2.dilate(im_thresh, np.ones((6, 6)))
#    im_thresh = cv2.erode(im_thresh, np.ones((6, 6)))

    im=-im
    th = cv2.GaussianBlur(im, (5, 5), 0)
    #plt_(th,'pp_gauss')
    th = cv2.threshold(th, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[1]
    #plt_(th,'pp_ttrunc')
    th = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #plt_(th,'pp_tbin')

    #small grouping
    th2 = cv2.dilate(th, np.ones((3,3)))
    th2 = cv2.erode(th2, np.ones((3,3)))
    #plt_(th2,'pp_small')

    #large grouping
    th = cv2.erode(th, np.ones((7, 7)))
    th = cv2.dilate(th, np.ones((7, 7)))
    #plt_(th2,'pp_large')

    th = -th

    filled = cv2.morphologyEx(
            th, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

    #plt_(filled,'pp_fill')

#    if interactive:
#        #cv2.namedWindow('blur')
#        #cv2.namedWindow('th_filled')
#        #cv2.imshow('blur', im_blur)
#        cv2.namedWindow('th')
#        cv2.imshow('th', filled)
        #cv2.imshow('th_filled', filled)
        #hist = cv2.calcHist([im], [0], None, [256], [0,256])
#    plt.plot(hist)
#    plt.axvline(x=tv, linewidth=2, color='k')
#    plt.xlim([0,256])
#    plt.title(tv)
#    plt.show()


    #im_thresh = cv2.adaptiveThreshold(
    #    im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #    t_blockSize, t_C
    #)

    return filled
