import logging
import os
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as p
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import rescale as skrescale

from gitter.utils import circularity
from .common import DEFAULT_FORMAT, FORMATS, GitterException
from .utils import set_contrast, autorotate_image, threshold_image, remove_rle, colony_peaks, round_odd, find_bounds

log = logging.getLogger(__name__)


class Gitter:
    data = None
    window = None
    thresholded = None
    plate_boundaries = None

    def __init__(self, image, options=None, **kwargs):
        """
        Constructor. Initialize with options or kwargs.

        :param image: Path to image to process.
        :param options: Preconfigured options instance, takes precedence over kwargs.
        :param kwargs: Override default options with keyword arguments (if "options" is not provided).
        :type image: str
        :type options: GitterOptions
        :type kwargs: keywords
        """
        self.path = image
        self.opt = options or GitterOptions(**kwargs)

    def load_image(self):
        """
        Opens the image and prepares it for processing (rescaling, auto-rotating,
        contrast, color inversion, thresholding).

        :return: Greyscale image as 2D float array with all preprocessing steps applied.
        :rtype: ndarray
        """
        im = imread(self.path, as_gray=True)

        # Rescale image for faster processing
        if self.opt.rescale:
            log.info('Rescaling image by %f' % (self.opt.rescale / im.shape[1],))
            im = skrescale(im, self.opt.rescale / im.shape[1], mode='reflect')

        # Try to automatically correct rotated plates.
        if self.opt.auto_rotate:
            im = autorotate_image(im)

        if self.opt.contrast:
            im = set_contrast(im, self.opt.contrast)

        if self.opt.inverse:
            im = 1 - im

        self.thresholded = threshold_image(im, self.opt.plate_rows)

        return im

    def grid(self):
        sum_cols, xlb, xrb = remove_rle(self.thresholded, p=0.3, axis=0)
        sum_rows, ylb, yrb = remove_rle(self.thresholded, p=0.3, axis=1)

        window_cols, col_peaks = colony_peaks(sum_cols, self.opt.plate_cols)
        window_rows, row_peaks = colony_peaks(sum_rows, self.opt.plate_rows)

        self.window = np.round(np.mean([window_cols, window_rows]))
        self.plate_boundaries = xlb, ylb, xrb, yrb

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

        new_columns = defaultdict(lambda: np.full(self.data.shape[0], np.nan))

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

            spot_bw_crop = self.thresholded[
                           int(rl + y):int(rr + y) - 1,
                           int(cl + x):int(cr + x) - 1]

            size = np.sum(spot_bw_crop)
            new_columns['size'][idx] = size
            new_columns['circularity'][idx] = circularity(spot_bw_crop, size)
            # self.data.loc[idx, 'size'] = np.sum(spot_bw_crop)
            # self.data.loc[idx, 'circularity'] = circularity(spot_bw_crop, self.data.loc[idx, 'size'])

            if self.opt.save_grid:
                # Store this only if we're saving grid
                self.data.loc[idx, 'rl'] = rl
                self.data.loc[idx, 'rr'] = rr
                self.data.loc[idx, 'cl'] = cl
                self.data.loc[idx, 'cr'] = cr

        self.data.loc[:, 'size'] = new_columns['size']
        self.data.loc[:, 'circularity'] = new_columns['circularity']

        return self.data

    def save(self):
        if self.opt.save_dat:
            dest = self.opt.save_dat
            if isinstance(dest, bool):
                dest = os.path.splitext(self.path)[0] + '.dat'

            with open(dest, 'w') as fout:
                if self.opt.colony_compat:
                    header = False

                    first = self.data.iloc[0]
                    last = self.data.iloc[-1]

                    fout.write("Colony Project Data File\n")
                    fout.write("%s\n" % self.path)
                    fout.write("image resolution:\n")
                    fout.write("%d %d\n" % self.thresholded.shape)
                    fout.write("Plate width and height( in CM):\n")
                    fout.write("12.700000 8.500000\n")
                    fout.write("Grid position(xmin, xmax, ymin, ymax):\n")
                    fout.write("%d %d %d %d\n" % (first.iloc[0], last.iloc[0], first.iloc[1], last.iloc[1]))
                    fout.write("Spot columns and rows, and total spots:\n")
                    fout.write("48 32 1536\n")
                    fout.write("First spot position:\n")
                    fout.write("%d %d\n" % tuple(first.iloc[:2]))
                    fout.write("rows  columns  size  circularity:\n")
                else:
                    header = True

                    fout.write("#Dat-format-version: 1\n")
                    fout.write("#Plate-boundaries: %d,%d,%d,%d\n" % self.plate_boundaries)
                    fout.write("#Grid-boundaries: %d,%d,%d,%d\n" % (tuple(self.data.iloc[0, :2]) + tuple(
                        self.data.iloc[-1, :2])))
                    fout.write("#Window: %d\n" % self.window)

            self.data.sort_values(['row', 'col']).to_csv(dest, sep='\t', index=False, header=header, mode='a',
                                                         columns=['row', 'col', 'size', 'circularity'])

        if self.opt.save_grid:
            dest = self.opt.save_grid
            if isinstance(dest, bool):
                dest = os.path.splitext(self.path)[0] + '_gridded.jpg'

            plt.imshow(self.thresholded, cmap='Greys_r')
            plt.gcf().set_size_inches((20, 10))

            for _, r in self.data.iterrows():
                plt.vlines([r.cl + r.x, r.cr + r.x], r.rl + r.y, r.rr + r.y, 'red')
                plt.hlines([r.rl + r.y, r.rr + r.y], r.cl + r.x, r.cr + r.x, 'red')

            plt.savefig(dest, dpi=200)
            plt.clf()

    @staticmethod
    def auto_process(image, **kwargs):
        gitter = Gitter(image, **kwargs)
        gitter.load_image()
        gitter.grid()
        gitter.quantify()
        gitter.save()

        return gitter


class GitterOptions:
    def __init__(self, plate_format=DEFAULT_FORMAT, remove_noise=False, auto_rotate=False, inverse=False,
                 contrast=None, rescale=0, save_grid=False, save_dat=True, colony_compat=False):
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

        if rescale and not (1500 <= rescale <= 4000):
            raise GitterException('Rescale width must be between 1500-4000px')

        self.plate_rows, self.plate_cols = plate_format
        self.remove_noise = remove_noise
        self.auto_rotate = auto_rotate
        self.inverse = inverse
        self.contrast = contrast
        self.rescale = rescale
        self.save_grid = save_grid
        self.save_dat = save_dat
        self.colony_compat = colony_compat
