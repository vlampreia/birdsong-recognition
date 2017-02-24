from scikits.audiolab import Sndfile
import operator
import os
from warnings import warn

def load_pcm(path):
    wave = Sndfile(path, "r")
    pcm = wave.read_frames(wave.nframes)
    wave.close()
    if wave.channels is not 1:
        pcm = pcm[:,0]
    return (pcm, wave.samplerate)


def list_types(path, types):
    paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in types:
                paths += [os.path.join(dirpath, filename)]
    return paths


def list_wavs(path):
    warn('list_wavs deprecated, use list_types instead')
    return list_types(path, ['.wav'])


def count_samples(samples_dir):
    species_count = {}
    paths = list_types(samples_dir, ['.mp3'])
    for path in paths:
        species = os.path.split(path)[0]
        species = os.path.split(species)[1]

        species_count[species] = species_count.get(species, 0) + 1

    #pprint.pprint(species_count, width=1)
    return species_count


def print_sample_statistics(samples_dir):
    samples = count_samples(samples_dir)
    s = sorted(samples.iteritems(), key=operator.itemgetter(1), reverse=True)

    fmt_samples = '{:>4}  {:<24}'

    print fmt_samples.format('WAVS','SAMPLES')
    print '{:_<80}'.format('')
    for sample in s:
        print fmt_samples.format(sample[1], sample[0][:24])

    print ''
    print sum(samples.values()), 'samples'
