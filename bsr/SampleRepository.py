from bsrdata import *
import futils
from IPython.core.debugger import Tracer
import pickle
from collections import defaultdict
import logging

class SampleRepository:
    samples = None
    spectrograms_dir = ''
    samples_dir = ''

    def __init__(self, spectrograms_dir, samples_dir):
        self.samples = []
        self.spectrograms_dir = spectrograms_dir
        self.samples_dir = samples_dir


    def filter_labels(self, labels, reject=False):
        if reject: self.samples = [s for s in self.samples if s.get_label() not in labels]
        else:      self.samples = [s for s in self.samples if s.get_label() in labels]


    def filter_uids(self, uids, reject=False):
        if reject: self.samples = [s for s in self.samples if s.get_uid() not in uids]
        else:      self.samples = [s for s in self.samples if s.get_uid() in uids]


    def reject_by_class_count(self, at_least=0, at_most=0):
        class_counts = defaultdict(int)

        for sample in self.samples:
            class_counts[sample.get_label()] += 1

        if at_least != 0: self.samples = [s for s in self.samples if class_counts[s.get_label()] >= at_least]
        if at_most != 0:  self.samples = [s for s in self.samples if class_counts[s.get_label()] <= at_most]

    def keep_n_of_each_class(self, n):
        selected_samples = []
        samples_selected_per_class = defaultdict(int)

        for sample in self.samples:
            if samples_selected_per_class[sample.get_label()] >= 20: continue

            selected_samples.append(sample)
            samples_selected_per_class[sample.get_label()] += 1

        self.samples = selected_samples

    def reject_by_template_count_per_class(self, at_least=0, at_most=0):
        template_counts_per_class = defaultdict(int)

        for sample in self.samples:
            template_counts_per_class[sample.get_label()] += len(sample.get_spectrogram().get_templates())

        if at_least != 0: self.samples = [s for s in self.samples if template_counts_per_class[s.get_label()] >= at_least]
        if at_most != 0:  self.samples = [s for s in self.samples if template_counts_per_class[s.get_label()] <= at_most]


    def gather_samples(self):
        paths = futils.list_files(self.spectrograms_dir, ['.png'])

        for path in paths:
            if 'templates' in path: continue
            self.samples.append(PersistentSample(path, self.spectrograms_dir))


    def get_samples_per_label(self):
        samples_per_label = defaultdict(list)

        for sample in self.samples:
            samples_per_label[sample.get_label()].append(sample)

        return samples_per_label


    def get_samples_by_uid(self, uids=[]):
        if uids is None: return []

        return [s for s in self.samples if s.get_uid() in uids]


    def get_all_templates(self, samples=None):
        if samples is not None:
            return [s.get_templates for s in samples]

        return [s.get_templates() for s in self.samples]


    def get_templates_by_uid(self, uids=[], samples=None):
        if uids is None: return []

        return [t for t in self.get_all_templates(samples) if t.get_uid() in uids]


    def store_all(self):
        for sample in samples.values():
            sample.store_all()


#    def load_spectrograms(self):
#        for sample in self.samples:
#            path = sample.get_spectrogram_path(self.spectrograms_dir)
#            if not os.path.exists(''.join([path, '.pkl'])): continue
#
#            sgram = Spectrogram.load_from_file(sample, path)
#
#            sample.spectrogram = sgram
#            num_spectrograms += 1
#
#
#    def load_templates(self):
#        for sample in self.samples:
#            if sample.spectrogram is None: continue
#
#            template_dir = sample.get_template_dir()


class PersistentSample(AbstractSample):
    _sample = None
    templates_dir = ''
    sample_dir = ''
    spectrogram_dir = ''
    _spectrograms_dir = ''


    def __init__(self, path, spectrograms_dir):
        self._spectrograms_dir = spectrograms_dir
        self._sample = self._from_path(path)


    def get_uid(self):
        return self._sample.get_uid();


    def get_label(self):
        return self._sample.get_label()


    def get_templates(self):
        return self.get_spectrogram().get_templates()


    def get_spectrogram(self):
        if self._sample.spectrogram is None:
            self._sample.spectrogram = PersistentSpectrogram(
                self,
                self.get_spectrogram_path()
            )

        return self._sample.spectrogram


    def get_pcm_path(self, samples_dir):
        if self.pcm_path == '':
            path = os.path.join(samples_dir, self.get_label())
            fname = self.get_uid() + '.wav'
            self.pcm_path = os.path.join(path, fname)

        return self.pcm_path


    def get_templates_path(self):
        if self.templates_dir == '':
            path = os.path.join(self._spectrograms_dir, self.get_label())
            self.templates_dir = os.path.join(path, 'templates')

        return self.templates_dir


    def get_spectrogram_path(self):
        if self.spectrogram_dir == '':
            path = os.path.join(self._spectrograms_dir, self._sample.get_label())
            self.spectrogram_dir = os.path.join(path, self._sample.get_uid())

        return self.spectrogram_dir


    def _from_path(self, path):
        if not os.path.exists(path): return None

        label = os.path.split(os.path.split(path)[0])[1]
        uid = os.path.splitext(os.path.split(path)[1])[0]

        return Sample(uid, label)


    def store_all(self):
        self._spectrogram.store_all()


class PersistentSpectrogram(AbstractSpectrogram):
    _spectrogram = None
    _path = ''
    _changed = False

    def __init__(self, sample, path):
        self._spectrogram = self.load_from_file(sample, path)
        self._path = path
        self._changed = False


    def get_pxx(self):
        return self._spectrogram.get_pxx()


    def get_freqs(self):
        return self._spectrogram.get_freqs()


    def get_times(self):
        return self._spectrogram.get_times()


    def get_label(self):
        return self._spectrogram.get_label()


    def get_templates(self):
        if self._spectrogram.templates == []:
            src_sample = self._spectrogram.src_sample
            template_dir = src_sample.get_templates_path()
            for f in os.listdir(template_dir):
                if not os.path.splitext(f)[1] == '.png': continue
                if not f.startswith(src_sample.get_uid()): continue

                t = PersistentTemplate(src_sample, os.path.join(template_dir, f))

                self._spectrogram.templates.append(t)

        return self._spectrogram.templates


    def load_from_file(self, src_sample, path):
        pxx = futils.image_load(path + '.png')
        with open(path + '.pkl', 'r') as f:
            freqs = pickle.load(f)
            times = pickle.load(f)

        return Spectrogram(src_sample, pxx, freqs, times)


    def write_to_file(self, path):
        dpath = os.path.split(path)[0]
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        success = futils.image_write(self.pxx, path + '.png')
        if success:
            with open(path + '.pkl', 'w') as f:
                pickle.dump(self.freqs, f)
                pickle.dump(self.times, f)


    def store_all(self):
        print 'check this '
        print 'check this '
        print 'check this '
        print 'check this '
        Tracer()()
        if self._changed: self.write_to_file(self._path)
        for template in self._spectrogram.get_templates():
            template.write_to_file()


class PersistentTemplate(AbstractTemplate):
    _template = None
    _path = ''


    def __init__(self, src_sample, path):
        self._template = self._load_from_file(src_sample, path)
        self._path = path


    def _load_from_file(self, src_sample, path):
        im = -futils.image_load(path)

        uid = os.path.splitext(os.path.split(path)[1])[0]
        idx = uid.split('-')[1]
        return Template(src_sample, im, uid, idx)


    def write_to_file(self, path):
        if self._has_changed: futils.image_write(self.get_im(), path)


    def get_uid(self):
        return self._template.get_uid()


    def get_idx(self):
        return self._template.get_idx()


    def get_im(self):
        return self._template.get_im()


    def get_src_sample(self):
        return self._template.get_src_sample()


