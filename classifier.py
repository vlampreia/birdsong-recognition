from scikits.audiolab import Sndfile
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import datetime
import pylab

path1 = "/home/victor/Downloads/woodwren1.wav"
path2 = "/home/victor/Downloads/woodwren.wav"

#def buildSpectrum(fpath):

def loadPcm(path):
    wave = Sndfile(path, "r")
    pcm = wave.read_frames(wave.nframes)
    wave.close()
    return (pcm, wave.samplerate)
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
    noverlap = nfft * 0.75
    vmin = None
    vmax = None
    
    pxx, freqs, times = matplotlib.mlab.specgram(
            pcm, NFFT=nfft, Fs=fs, noverlap=noverlap
            )

    return (pxx, freqs, times)

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

def filterSpecgram(pxx, freqs, times, threshold):
    median_times = np.median(pxx, axis=0)
    median_freqs = np.median(pxx, axis=1)

    fpxx = pxx.copy();
    for i in range(fpxx[:,0].size):
        for j in range(fpxx[0,:].size):
            fpxx[i,j] = (fpxx[i,j] > median_times[j]*threshold and
                         fpxx[i,j] > median_freqs[i]*threshold)
    return fpxx

#buildSpectrum("/home/victor/Downloads/woodwren.wav")
