'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
import requests
import urllib

import numpy as np
import hickle as hkl

from bs4 import BeautifulSoup

# Check scipy version for deprecated imread
from scipy import __version__ as scipy_version
def _use_skimage():
    from skimage.io import imread
    from skimage.transform import resize as imresize
    return imread, imresize
if scipy_version >= '1.2.0':
    imread, imresize = _use_skimage()
else:
    try:
        # I think there's an issue in Anaconda that prevents it from
        # installing the proper submodules of scipy.
        from scipy.misc import imread, imresize
    except ImportError:
        imread, imresize = _use_skimage()

from kitti_settings import DATA_DIR

def _vprint(verbose, string):
    if verbose:
        print(string)


desired_im_sz = (128, 160)
categories = [
    'city',
    'residential', 
    'road'
    ]

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
val_recordings = [('city', '2011_09_26_drive_0005_sync')]
test_recordings = [
    ('city', '2011_09_26_drive_0104_sync'),
    ('residential', '2011_09_26_drive_0079_sync'), 
    ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): 
    os.mkdir(DATA_DIR)


# Download raw zip files by scraping KITTI website
def download_data(verbose=False, skip_downloaded=False):
    base_dir = os.path.join(DATA_DIR, 'raw') + os.sep
    if not os.path.exists(base_dir): 
        os.mkdir(base_dir)

    for c in categories:
        url = 'http://www.cvlibs.net/datasets/kitti/raw_data.php?type=' + c
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        drive_list = soup.find_all('h3')
        drive_list = [d.text[:d.text.find(' ')] for d in drive_list]

        _vprint(verbose, 'Downloading set: ' + c)
        
        c_dir = base_dir + c + os.sep
        if not os.path.exists(c_dir): 
            os.mkdir(c_dir)

        for i, d in enumerate(drive_list):
            _vprint(verbose, str(i+1) + os.sep + str(len(drive_list)) + ": " + d)
            
            # Old: http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
            # New: https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip
            url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/' + d + '/' + d + '_sync.zip'
            _vprint(verbose, "url: " + url)

            output_file = os.path.join(c_dir, d + '_sync.zip')
            print("output_file", output_file)
            if os.path.exists(output_file) and skip_downloaded:
                _vprint(verbose, "File already downloaded, skipping.")
                continue
        
            # curl -L <url> -o --create-dirs ./kitti_data/raw/<category>/<name>.zip
            # os.system('curl -L ' + url + ' -o ' + os.path.join(c_dir, d + '_sync.zip') + ' --create-dirs')

            # Alternatively: mkdir ./kitti_data/raw/<category>/
            #                wget -O ./kitti_data/raw/<category>/<name>.zip <url>
            download_to_folder = os.path.join(c_dir, d + '_sync.zip')
            os.system('mkdir ' + download_to_folder)
            os.sytem('wget -O ' + download_to_folder + url)


# unzip images
def extract_data(verbose=False, stop_short=False):
    _vprint(verbose, 'For c in categories...')

    error = 0
    for c in categories:
        if error and stop_short:
            _vprint(verbose, 'Received exit code ' + str(error))
            print('Exiting...')
            break
        
        _vprint(verbose, '\t' + 'category: +' + c)

        c_dir = os.path.join(DATA_DIR, 'raw', c) + os.sep
        _vprint(verbose, '\t' + 'c_dir: + ' + c_dir)

        _, _, zip_files = os.walk(c_dir).next()
        _vprint(verbose, '\t' + 'Found zip-files. For f in zip_files...')

        for f in zip_files:
            _vprint(verbose, '\t' * 2 + 'Unpacking: ' + f)            

            spec_folder = os.path.join(f[:10], f[:-4], 'image_03', 'data*')
            command = 'unzip -qq ' + c_dir + f + ' ' + spec_folder + ' -d ' + c_dir + f[:-4]
            _vprint(verbose, '\t' * 2 + 'Executing: ' + command)
                
            error = os.system(command)
            if error and stop_short:
                break
        print('\n')


# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data(verbose=False):
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings

    not_train = splits['val'] + splits['test']

    # Randomly assign recordings to training and testing. 
    # Cross-validation done across entire recordings.
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw', c) + os.sep
        _, folders, _ = os.walk(c_dir).next()
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from

        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw', category, folder, folder[:10], folder, 'image_03', 'data')
            _, _, files = os.walk(im_dir).next()
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')

        # Michael Ala: X is an awful name, why not just "image_data"?

        # X is an array of image data, each image being a 3-dimensional array of
        # dimensions (img_width) x (img_height) x 3 (colour channels).
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            # im array info
            #   First axis: rows
            #   Second axis: columns
            #   Third axis: colour
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

        # For future reference, the way this works is that the image at index i
        # in X corresponds to the source at index i in the source_list. Also, the
        # source is the name of the category (i.e. "drive", "residential", or "road"),
        # and the name of the folder, separated by a dash. For example,
        # road-2011_09_26_drive_0027_sync


# resize and crop image
def process_im(im, desired_sz):

    # Note: the given images are landscape, so we can always safely scale
    # along the y-axis and crop the x-axis.
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    
    # Since the first axis of the image array contains the rows of the image,
    # to crop along the x-axis we restrict the index of the second dimension.
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    
    # Michael Ala 6/18/2018: Added some command line inputs for debugging purposes.
    import sys
    args = sys.argv

    no_download     = "--no-download"     in args
    no_extract      = "--no-extract"      in args
    verbose         = "--verbose"         in args
    stop_short      = "--stop-short"      in args
    skip_downloaded = "--skip-downloaded" in args

    if not no_download:
        download_data(verbose=verbose, skip_downloaded=skip_downloaded)
    if not no_extract:
        extract_data(verbose=verbose, stop_short=stop_short)
    process_data(verbose=verbose)
