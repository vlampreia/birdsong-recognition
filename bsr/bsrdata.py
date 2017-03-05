import os

class Sample:
    uid = ''
    label = ''
    spectrogram = None

    def __init__(self, uid, label):
        self.uid = uid
        self.label = label

    def get_pcm_path(self, samples_dir):
        fname = ''.join([self.uid, '.wav'])
        return os.path.join(os.path.join(samples_dir, self.label), fname)

    def get_spectrogram_path(self, spectrograms_dir):
        return os.path.join(os.path.join(spectrograms_dir, self.label), self.uid)

    def get_template_dir(self, spectrograms_dir):
        return os.path.join(os.path.join(spectrograms_dir, self.label), 'templates')

    def get_spectrogram(self):
        return spectrogram

    def get_templates(self):
        return self.spectrogram.templates


class Spectrogram:
    src_sample = None
    pxx = None
    freqs = None
    times = None
    templates = None


    def __init__(self, sample, pxx, freqs, times):
        self.src_sample = sample
        self.pxx = pxx
        self.freqs = freqs
        self.times = times
        self.templates = []


class Template:
    idx = -1
    uid = ''
    src_sample = None
    im = None

    def __init__(self, src, template, idx):
        self.src_sample = src
        self.im = template
        self.idx = idx
        self.uid = ''.join([self.src_sample.uid, '-', str(self.idx)])
