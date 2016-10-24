from scikits.audiolab import Sndfile
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import datetime
import pylab
import cv2

#def buildSpectrum(fpath):

#pcmf = [float(val) / pow(2, 15) for val in pcm]
#fs = wave.samplerate
#nfft = int(fs*0.005)
#nfft = 512
#noverlap = nfft * 0.75#int(fs * 0.75)
#framesize = 0.050
#framehop = 0.025
#framesamples = int(framesize * samplerate)
#hopsamples = int(framehop * samplerate)
#hwindow = scipy.hanning(framesamples)
#X = scipy.array([scipy.fft(hwindow*pcmf[i:i+framesamples]) for i in range(0, len(pcmf)-framesamples, hopsamples)])
#vmin = None #db threshold
#vmax = None


# plt.specgram computes spectrogram for us..
#fig, ax = plt.subplots()
#pxx, freqs, times, im = ax.specgram(
#        pcm, NFFT=nfft, Fs=fs, noverlap=noverlap,
#        vmin=vmin, vmax=vmax,
#        cmap=pylab.get_cmap('Greys'))
#
#cbar = fig.colorbar(im)
#cbar.set_label('dB')
#ax.axis('tight')
#
#ax.set_xlabel('time h:mm:ss')
#ax.set_ylabel('kHz')
#
#scale = 1e3
#ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
#ax.yaxis.set_major_formatter(ticks)

def loadPcm(path):
    wave = Sndfile(path, "r")
    pcm = wave.read_frames(wave.nframes)
    wave.close()
    return (pcm, wave.samplerate)

#-------------------------------------------------------------------------------

def timeTicks(x, pos):
    d = datetime.timedelta(seconds=x)
    return str(d)
#formatter = matplotlib.ticker.FuncFormatter(timeTicks);
#ax.xaxis.set_major_formatter(formatter)

#plt.show()

#-------------------------------------------------------------------------------

def makeSpecgram(pcm, samplerate):
    fs = samplerate
    nfft = 512
    window = np.hamming(512)
    noverlap = 512 * 0.75
    vmin = None
    vmax = None
    
    pxx, freqs, times = matplotlib.mlab.specgram(
            pcm, NFFT=nfft, Fs=fs, noverlap=noverlap, window=window
            )

    return (pxx, freqs, times)

#-------------------------------------------------------------------------------

def plotSpecgram(pxx, freqs, times, log=False):
    fig, ax = plt.subplots()
    norm = None
    if log:
        norm = LogNorm()

    extent = [times.min(), times.max(), freqs.min(), freqs.max()]
    im = ax.imshow(pxx, extent=extent, origin='lower', aspect='auto',
            cmap=pylab.get_cmap('Greys'), norm=norm
            )
#    im = ax.pcolormesh(times, freqs, 10 * np.log10(pxx),
 #           cmap=pylab.get_cmap('Greys'))
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

def plotMultiple(graphs, window):
    fig, ax = plt.subplots(len(graphs), 1)
    for i in range(len(graphs)):
        ax[i].imshow(
            graphs[i][window[0][0]:window[0][1],window[1][0]:window[1][1]],
            cmap=pylab.get_cmap('Greys_r')
        );
    fig.show()

#-------------------------------------------------------------------------------

def filter_specgram(im):
    t_blockSize = 11
    t_C = 5
    b_kernelsize = 5

    im_blur = cv2.medianBlur(im, b_kernelsize)
    im_thresh = cv2.adaptiveThreshold(
        im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        t_blockSize, t_C
    )

    im_result = im_thresh
    return im_result

def write_specgram(pxx, fname):
    matplotlib.image.imsave(
        fname,
        10*np.log10(pxx),
        origin='lower',
        cmap=pylab.get_cmap('Greys')
    )

def load_specgram(fname):
    im = cv2.imread(fname, 0)
    return im


#
# Example usage pattern:
#   >>> pcm, fs = loadPcm('./sample.wav')
#   >>> pxx, freqs, times = makeSpecgram(pxx, freqs, times, True)
#   >>> write_specgram(pxx, './sample.png')
#   >>> im = load_specgram('./sample.png')
#   >>> im_ = filter_specgram(im)

#buildSpectrum("/home/victor/Downloads/woodwren.wav")

