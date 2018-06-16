import os
import cv2
import matplotlib
import pylab
import logging

def image_load(path):
    if not os.path.exists(path): return None

    return cv2.imread(path, 0)


def image_write(im, path):
    try:
        print(path)
        matplotlib.image.imsave(
            path,
            im,
            origin='lower',
            cmap=pylab.get_cmap('Greys')
        )
    except:
        logging.error('unexpected error writing image to file {}'.format(path))
        return False

    return True

def list_files(path, extensions):
    paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in extensions:
                paths.append(os.path.join(dirpath, filename))

    return paths

def pickle_safe(data, path):
    def _mv_bak(path):
        new_path = path + '.bak'
        if os.path.exists(new_path):
            _mv_bak(new_path)
        os.rename(path, new_path)

    if os.path.exists(path):
        _mv_bak(path)

    with open(path, 'w') as f:
        pickle.dump(results, f)
