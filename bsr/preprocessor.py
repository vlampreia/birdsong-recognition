import utils
import specgram_utils

import os
import numpy as np
import cv2


def process(pcm_dir, specgram_dir):
    pcm_paths = utils.list_wavs(pcm_dir)
    spec_paths = specgram_utils.create_specgrams(pcm_paths, specgram_dir, False)
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

def extract_features(spec_paths, features_dir):
    for path in spec_paths:
        print 'extracting templates from', path
        if not os.path.exists(path):
            print 'ERROR: path doesn\'t exist:', path, 'skipping..'
            continue;

        im = specgram_utils.load_specgram(path)
        features = extract_templates(im)

        filename = os.path.splitext(os.path.split(path)[1])[0]
        fpath = os.path.split(path)[0]
        fpath = os.path.join(fpath, features_dir)
        if not os.path.exists(fpath): os.makedirs(fpath)
        fpath = os.path.join(fpath, filename + '-')

        for i in xrange(len(features)):
            cv2.imwrite(fpath + str(i) + '.png', -features[i])


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


def extract_templates(im):
    """
    Extract all templates from a given spectrogram image
    """

#    tmp = cv2.medianBlur(im, 5)
#    tmp = cv2.threshold(tmp, 255*0.65, 255, cv2.THRESH_BINARY)[1]
    tmp = im.copy()
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
