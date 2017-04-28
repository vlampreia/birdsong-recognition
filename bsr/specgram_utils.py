from utils import *

import cv2
import numpy as np
import matplotlib
import pylab
import os
import sys
import pickle


def load_specgram(path):
    sgram = cv2.imread(''.join([path, '.png']), 0)

    with open(''.join([path, '.pkl']), 'r') as f:
        freqs = pickle.load(f)
        times = pickle.load(f)

    return (sgram, freqs, times)

def write_specgram(pxx, freqs, times, path):
    dpath = os.path.split(path)[0]
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    try:
        matplotlib.image.imsave(
            ''.join([path, '.png']),
            pxx,
            origin='lower',
            cmap=pylab.get_cmap('Greys')
            )

        with open(''.join([path, '.pkl']), 'w') as f:
            pickle.dump(freqs, f)
            pickle.dump(times, f)

    except IOError as err:
        print '\terror writing specgram file: {}'.format(path), err
    except NameError as err:
        print '\terror writing specgram file: {}'.format(path), err
    except ValueError as err:
        print '\terror writing specgram file: {}'.format(path), err
    except:
        print '\terror writing specgram file: {}'.format(path), sys.exc_info()[0]

def make_specgram(pcm, samplerate):
    fs = samplerate
    nfft = 256
    window = np.hamming(nfft)
    noverlap = nfft * 0.75
    vmin = None
    vmax = None

    min_freq = 100
    max_freq = 10000

    pxx, freqs, times = matplotlib.mlab.specgram(
            pcm, NFFT=nfft, Fs=fs, noverlap=noverlap, window=window
            )

    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    pxx = pxx[freq_mask,:]

    pxx = 10*np.log10(pxx.clip(min=0.0000000001))
    pxx = np.array(pxx, dtype=np.uint8)

    freqs = freqs[freq_mask]

    return (pxx, freqs, times)

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

        pcm, fs = load_pcm(path)
        pxx, freqs, times = make_specgram(pcm, fs)
        write_specgram(pxx, spath)
    return paths
