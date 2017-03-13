import os

class AbstractSample(object):

    def get_uid(self):
        raise NotImplementedError()

    def get_label(self):
        raise NotImplementedError()

    def get_templates(self):
        raise NotImplementedError()


    def get_spectrogram(self):
        raise NotImplementedError()


class Sample(AbstractSample):
    uid = ''
    label = ''
    spectrogram = None

    def __init__(self, uid, label):
        self.uid = uid
        self.label = label


    def get_spectrogram(self):
        return self.spectrogram


    def get_templates(self):
        if self.spectrogram is None: return []
        return self.spectrogram.templates

    def get_uid(self):
        return self.uid

    def get_label(self):
        return self.label


class AbstractSpectrogram(object):
    def get_pxx(self):
        raise NotImplementedError()

    def get_freqs(self):
        raise NotImplementedError()

    def get_times(self):
        raise NotImplementedError()

    def get_templates(self):
        raise NotImplementedError()

    def get_label(self):
        raise NotImplementedError()


class Spectrogram(AbstractSpectrogram):
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


    def get_pxx(self):
        return self.pxx

    def get_freqs(self):
        return self.freqs

    def get_times(self):
        return self.times

    def get_templates(self):
        return this.templates

    def get_label(self):
        return self.src_sample.get_label()


class AbstractTemplate(object):

    def get_src_sample(self):
        raise NotImplementedError()

    def get_idx(self):
        raise NotImplementedError()

    def get_uid(self):
        raise NotImplementedError()

    def get_im(self):
        raise NotImplementedError()


class Template(AbstractTemplate):
    src_sample = None
    idx = -1
    uid = ''
    im = None

    def __init__(self, src, template, uid, idx):
        self.src_sample = src
        self.im = template
        self.idx = idx
        self.uid = uid
        #self.uid = ''.join([self.src_sample.uid, '-', str(self.idx)])

    def get_src_sample(self):
        return self.src_sample

    def get_idx(self):
        return self.idx

    def get_uid(self):
        return self.uid

    def get_im(self):
        return self.im
