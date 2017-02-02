import utils
import specgram_utils

import os
import numpy as np
import cv2


def extract_templates(im):
    """
    Extract all templates from a given spectrogram image
    """

#    tmp = cv2.medianBlur(im, 5)
#    tmp = cv2.threshold(tmp, 255*0.65, 255, cv2.THRESH_BINARY)[1]
    tmp = filter_specgram(im)
    _, contours, _ = cv2.findContours(
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

        im = specgram_utils.load_specgram(path)
        im_ = filter_specgram(im)
        cv2.imwrite(fpath, im_)
