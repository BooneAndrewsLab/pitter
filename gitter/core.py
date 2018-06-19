import logging
from itertools import product

import numpy as np
import pandas as p
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import rescale

from .common import DEFAULT_FORMAT, FORMATS, GitterException
from .utils import set_contrast, autorotate_image, threshold_image, remove_rle, colony_peaks, round_odd, \
    find_bounds

log = logging.getLogger(__name__)


class Gitter:
    im = None
    thresholded = None
    data = None
    window = None

    def __init__(self, image, options=None, **kwargs):
        self.path = image
        self.opt = options or GitterOptions(**kwargs)

    def load_image(self):
        im = imread(self.path, as_grey=True)

        if self.opt.fast:
            log.info('Rescaling image by %f' % (self.opt.fast / im.shape[1],))
            im = rescale(im, self.opt.fast / im.shape[1], mode='reflect')

        if self.opt.auto_rotate:
            im = autorotate_image(im)

        if self.opt.contrast:
            im = set_contrast(im, self.opt.contrast)

        if self.opt.inverse:
            im = 1 - im

        self.im = im
        self.thresholded = threshold_image(im, self.opt.plate_rows)

        return self.im

    def grid(self):
        sum_cols, xlb, xrb = remove_rle(self.thresholded, p=0.6, axis=0)
        sum_rows, ylb, yrb = remove_rle(self.thresholded, p=0.6, axis=1)

        window_cols, col_peaks = colony_peaks(sum_cols, self.opt.plate_cols)
        window_rows, row_peaks = colony_peaks(sum_rows, self.opt.plate_rows)

        self.window = np.round(np.mean([window_cols, window_rows]))

        self.data = p.DataFrame(list(product(col_peaks, row_peaks)), columns=['x', 'y'])
        self.data = self.data.merge(
            p.DataFrame(np.arange(1, col_peaks.shape[0] + 1), np.sort(col_peaks), columns=['col']),
            how='left', left_on='x', right_index=True)
        self.data = self.data.merge(
            p.DataFrame(np.arange(1, row_peaks.shape[0] + 1), np.sort(row_peaks), columns=['row']),
            how='left', left_on='y', right_index=True)

    def quantify(self):
        colony_eps = round_odd(np.round(self.window * 1.5)) / 2
        minb = np.round(colony_eps / 3)
        sizes = []

        if self.opt.save_grid:
            plt.imshow(self.thresholded, cmap='Greys_r')
            plt.gcf().set_size_inches((20, 10))

        for idx, x, y, ccolumn, crow in self.data.itertuples():
            cent_pixel = self.thresholded[y, x]

            if not cent_pixel:
                # Look for colony in near vicinity
                cent_pixel = self.thresholded[y - 10:y + 11, x - 10:x + 11]

                if np.sum(cent_pixel):
                    col_dist = np.argwhere(cent_pixel == 1) - (10, 10)
                    closest_pix = col_dist[np.argmax((col_dist ** 2).sum(1))]

                    x += closest_pix[1]
                    y += closest_pix[0]

            rect = list(map(int, [x - colony_eps, x + colony_eps, y - colony_eps, y + colony_eps]))
            spot_bw = self.thresholded[rect[2]:rect[3], rect[0]:rect[1]]

            if np.sum(cent_pixel):
                rs = np.sum(spot_bw, axis=1)
                cs = np.sum(spot_bw, axis=0)

                rl, rr = find_bounds(rs)
                cl, cr = find_bounds(cs)
            else:
                rl, rr = -minb, minb
                cl, cr = -minb, minb

            if self.opt.save_grid:
                plt.vlines([cl + x, cr + x], rl + y, rr + y, 'red')
                plt.hlines([rl + y, rr + y], cl + x, cr + x, 'red')

            sizes.append(np.sum(self.thresholded[
                                int(rl + y):int(rr + y) - 1,
                                int(cl + x):int(cr + x) - 1]))

        if self.opt.save_grid:
            plt.savefig('./test.png', dpi=200)

        self.data.loc[:, 'size'] = sizes

        print(self.data)
        return self.data

    @staticmethod
    def auto_process(image, options=None, **kwargs):
        gitter = Gitter(image, options, **kwargs)
        gitter.load_image()
        gitter.grid()
        gitter.quantify()

        return gitter


class GitterOptions:
    def __init__(self, plate_format=DEFAULT_FORMAT, remove_noise=False, auto_rotate=False, inverse=False,
                 contrast=None, fast=0, save_grid=False, save_dat=True):
        # Check if we have one number plate formats
        if isinstance(plate_format, int):
            if plate_format not in FORMATS:
                raise GitterException('''Invalid plate density, please use 1536, 384 or 96. If the density of your plate is 
        not listed, you can specifcy a tuple of the number of rows and columns in your plate (e.g. (32,48))''')
            plate_format = FORMATS[plate_format]

        if not isinstance(plate_format, tuple) and len(plate_format) != 2:
            raise GitterException('''Invalid plate format, plate formats must be a tuple of the number of rows and columns 
    (e.g. (32,48)) or a value indicating the density of the plate (e.g 1536, 384 or 96) possible''')

        if contrast and contrast <= 0:
            raise GitterException('Contrast value must be positive')

        if fast and not (1500 <= fast <= 4000):
            raise GitterException('Fast resize width must be between 1500-4000px')

        self.plate_rows, self.plate_cols = plate_format
        self.remove_noise = remove_noise
        self.auto_rotate = auto_rotate
        self.inverse = inverse
        self.contrast = contrast
        self.fast = fast
        self.save_grid = save_grid
        self.save_dat = save_dat
