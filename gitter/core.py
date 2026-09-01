import logging
import os
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as p
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from skimage.io import imread
from skimage.transform import rescale as skrescale

from gitter.utils import circularity
from .common import DEFAULT_FORMAT, FORMATS, GitterException
from .utils import set_contrast, autorotate_image, threshold_image, remove_rle, colony_peaks, round_odd, find_bounds

log = logging.getLogger(__name__)

#: How the gridded overlay is written. Half resolution still leaves roughly 40 pixels per
#: colony on a 1536 plate, which is more than enough to see a grid that has slipped, and
#: it is the difference between 1.4MB and 0.5MB per plate -- a 40 plate screen is 57MB of
#: debugging aid at full size. 100 dpi with the figure sized in pixels/100 means one
#: figure inch is one hundred image pixels, so the output is the source image's own
#: dimensions scaled by GRID_SCALE and nothing else.
GRID_SCALE = .5
GRID_QUALITY = 85
GRID_DPI = 100


class Gitter:
    data = None
    window = None
    original_image = None
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

        self.original_image = im
        self.thresholded = threshold_image(im, self.opt.plate_rows)

        return im

    def grid(self):
        ref_plate = self.opt.template

        if ref_plate and self.opt.use_template_locations:
            self.window = ref_plate.window
            self.plate_boundaries = ref_plate.plate_boundaries
            self.data = ref_plate.data.copy()
            # print(self.data.loc[self.data.col.isin([16, 17])])
        else:
            # plt.imshow(self.thresholded, cmap='Greys_r')
            # plt.savefig('/home/matej/tet.png')

            sum_cols, xlb, xrb = remove_rle(self.thresholded, p=0.2, axis=0,
                                            override_boundaries=ref_plate.plate_boundaries[::2] if ref_plate else None)
            sum_rows, ylb, yrb = remove_rle(self.thresholded, p=0.2, axis=1,
                                            override_boundaries=ref_plate.plate_boundaries[1::2] if ref_plate else None)

            # plt.plot(sum_cols)
            # plt.savefig('/home/matej/tet.png')

            window_cols, col_peaks = colony_peaks(sum_cols, self.opt.plate_cols, self.opt.border_to_zero)

            # plt.imshow(self.thresholded, cmap='Greys_r')
            # plt.vlines(col_peaks, 0, 3000)
            # plt.savefig('/home/matej/tet.png')

            window_rows, row_peaks = colony_peaks(sum_rows, self.opt.plate_rows, self.opt.border_to_zero)

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
        if self.opt.liquid_assay:
            return self.quantify_liquid()
        return self.quantify_solid()

    def quantify_liquid(self):
        colony_grid_im = self.original_image[
                         self.plate_boundaries[1]:self.plate_boundaries[3],
                         self.plate_boundaries[0]:self.plate_boundaries[2]
                         ]

        # 255 bins for 255 pixel intensities (8bit image)
        counts, bins = np.histogram(colony_grid_im.flatten(), bins=255, range=(0, 1))
        global_background = bins[counts.argmax() + 1]

        # We'll look at 80% of max window
        colony_eps = round_odd(self.window * 0.9) // 2

        new_columns = defaultdict(lambda: np.full(self.data.shape[0], np.nan))

        # Use template's coordinates
        for idx, x, y, ccolumn, crow in self.opt.template.data.itertuples():
            spot_gray_crop = self.original_image[
                             y - colony_eps:y + colony_eps,
                             x - colony_eps:x + colony_eps
                             ]

            background = global_background
            if self.opt.local_illumination:
                """
                Get local background value. Local background is the most common pixel value in the area of 5 colonies
                in all directions. This avoids illumination gradient caused by the lamp in the imager. The centre of 
                the image is usually brighter than the sides.
                WARNING: this is really slow!
                TODO: clip the cropped window to plate boundaries. Ie: don't allow neighborhood_im to see plate borders
                """
                neighborhood_eps = int(self.window * 5)
                neighborhood_im = self.original_image[
                                  y - neighborhood_eps:y + neighborhood_eps,
                                  x - neighborhood_eps:x + neighborhood_eps
                                  ]
                # 255 bins for 255 pixel intensities (8bit image)
                counts, bins = np.histogram(neighborhood_im.flatten(), bins=255, range=(0, 1))
                background = bins[counts.argmax() + 1]

            # Subtract background to normalize, clip to 0
            spot_gray_crop = np.clip(spot_gray_crop - background, 0, 1)

            size = np.sum(spot_gray_crop)
            new_columns['size'][idx] = size
            new_columns['circularity'][idx] = 0

        self.data.loc[:, 'size'] = new_columns['size']
        self.data.loc[:, 'circularity'] = new_columns['circularity']

        return self.data

    def quantify_solid(self):
        colony_eps = int(round_odd(np.round(self.window * 1.5)) / 2)
        minb = int(np.round(colony_eps * .6))

        new_columns = defaultdict(lambda: np.full(self.data.shape[0], np.nan))

        for idx, x, y, ccolumn, crow in self.data.itertuples():
            cent_pixel = self.thresholded[y, x]

            if not cent_pixel:
                # Try to recenter in a 20x20 neighborhood, don't increase the box
                # The colony might be just small and off to one side. Or the pin left a "crater" and we have an atoll
                # shape, a spotty colony
                cent_pixel = self.thresholded[y - 10:y + 10 + 1, x - 10:x + 10 + 1]

                if np.sum(cent_pixel):
                    col_dist = np.argwhere(cent_pixel == 1) - (10, 10)
                    closest_pix = col_dist[np.argmax((col_dist ** 2).sum(1))]

                    x += closest_pix[1]
                    y += closest_pix[0]

            # First check
            spot_bw_crop = self.thresholded[
                           y - minb:y + minb,
                           x - minb:x + minb
                           ]

            # np.save('/home/matej/kalb/cutouts/%s_%s.dat' % (ccolumn, crow), spot_bw_crop)

            tigh_ratio = np.sum(spot_bw_crop) / (spot_bw_crop.shape[0] ** 2)
            if tigh_ratio < .1:
                # Only 5% is a potential colony
                rl, rr = -minb, minb
                cl, cr = -minb, minb
            else:
                spot_bw = self.thresholded[
                          y - colony_eps:y + colony_eps,
                          x - colony_eps:x + colony_eps
                          ]

                # plt.imshow(spot_bw)
                # plt.savefig('/home/matej/kalb/cutouts/%s_%s.jpg' % (ccolumn, crow))
                # plt.clf()

                if np.sum(cent_pixel):
                    rs = p.Series(np.sum(spot_bw, axis=1))
                    cs = p.Series(np.sum(spot_bw, axis=0))

                    rl, rr = find_bounds(rs.values)
                    cl, cr = find_bounds(cs.values)
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

                self.data.loc[idx, 'x'] = x
                self.data.loc[idx, 'y'] = y
                self.data.loc[idx, 'newx'] = x
                self.data.loc[idx, 'newy'] = y

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

            self.save_grid_image(dest)

    def save_grid_image(self, dest):
        """
        Draw the fitted grid over the plate and write it out.

        Three things this does not do, each of which used to cost more than the
        measurement it illustrates. It does not create four artists per colony -- 6144 of
        them took 4.6s where one LineCollection takes 0.06s. It does not render a fixed
        40x20 inches at 200 dpi -- that is an 8000x4000 canvas for a 3888x2592 photograph,
        5.8s of rendering to upscale an image and surround it with white margin. And it
        does not go through pyplot, so there is no global figure to remember to close and
        no backend for a worker process to guess at.

        :param dest: Path to write the JPEG to.
        :type dest: str
        """
        base = self.thresholded if self.opt.grid_on_thresholded else self.original_image
        height, width = base.shape

        # One closed path per colony, all in a single collection.
        d = self.data
        left, right = d.cl + d.x, d.cr + d.x
        top, bottom = d.rl + d.y, d.rr + d.y
        boxes = [[(l, t), (r, t), (r, b), (l, b), (l, t)]
                 for l, r, t, b in zip(left, right, top, bottom)]

        figure = Figure(figsize=(width / GRID_DPI * GRID_SCALE,
                                 height / GRID_DPI * GRID_SCALE), dpi=GRID_DPI)
        FigureCanvasAgg(figure)

        # add_axes over the whole figure, and no axis: the ticks were labelling pixel
        # coordinates of a plate photograph, which nobody has ever needed.
        ax = figure.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.imshow(base, cmap='Greys_r', interpolation='nearest', aspect='auto')
        ax.add_collection(LineCollection(boxes, colors='red', linewidths=.7))
        ax.plot(d.x, d.y, 'b.', markersize=1)
        ax.plot(d.newx, d.newy, 'y.', markersize=1)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        figure.savefig(dest, dpi=GRID_DPI, pil_kwargs={'quality': GRID_QUALITY})

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
                 contrast=None, rescale=0, save_grid=False, save_dat=True, colony_compat=False, template=None,
                 template_plate=None, ignore_errors=False, resume_processing=False, liquid_assay=False,
                 local_illumination=False, zero_border=False, use_template_locations=False,
                 grid_on_thresholded=False):
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

        rescale = int(rescale)

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
        self.template = template
        self.template_plate = template_plate
        self.ignore_errors = ignore_errors
        self.resume_processing = resume_processing
        self.liquid_assay = liquid_assay
        self.local_illumination = local_illumination
        self.border_to_zero = zero_border
        self.use_template_locations = use_template_locations
        # Draw the grid over the black-and-white mask instead of the plate. Useful when
        # what you are debugging is the thresholding itself; the default draws over the
        # photograph, where a grid that fitted the mask but not the colonies shows up.
        self.grid_on_thresholded = grid_on_thresholded
