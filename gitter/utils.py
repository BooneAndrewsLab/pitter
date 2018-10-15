import logging

import numpy as np
from scipy import signal
from skimage.measure import perimeter
from skimage.morphology import opening, square
from skimage.transform import radon, rescale, rotate, resize

log = logging.getLogger(__name__)


def set_contrast(im, contrast=10):
    """
    Adjust contrast of an image.

    :param im: image
    :param contrast: Adjustment to the contrast
    :type im: ndarray
    :type contrast: int

    :return: Image with adjusted contrast
    :rtype: ndarray
    """
    c = (100.0 + contrast) / 100.0
    im = ((im - 0.5) * c) + 0.5
    return np.clip(im, 0, 1)


def round_odd(x):
    """
    Round up to the closest odd integer.

    :param x: number to round
    :type x: [int, float]

    :return: Number rounded up to the closest odd integer.
    :rtype: int
    """
    x = np.floor(x)
    return int(x + 1 - x % 2)


def _find_rotation_angle(im, degree_inc=0.2, scale_to_h=500, angle_eps=50):
    im = rescale(im, scale_to_h / im.shape[0], mode='reflect', multichannel=False, anti_aliasing=True)
    samples = (angle_eps * 2) / degree_inc

    theta = np.linspace(90 - angle_eps, 90 + angle_eps, int(samples), endpoint=False)
    sinogram = radon(im, theta=theta, circle=True)

    row_var = np.apply_along_axis(np.var, 0, sinogram)

    peaks, = signal.argrelmax(row_var, order=int(10 / degree_inc))
    peak = peaks[np.argmin(np.abs(peaks - (samples / 2)))]

    return ((samples / 2) - peak) * degree_inc


def autorotate_image(im, **kwargs):
    """
    Detect if the image is rotated and correct it to be horizontal.

    :param im: image
    :param kwargs: additional arguments passed along to _find_rotation_angle.
    :type im: ndarray
    :type kwargs: keywords

    :return: Rotated image
    :rtype: ndarray
    """
    angle = _find_rotation_angle(im, **kwargs)
    log.info('Rotate for %.2f' % (angle,))
    return rotate(im, angle)


def _find_optimal_threshold(x, r=2, lim=(0, 0.4), cap=0.2):
    x[(x < lim[0]) | (x > lim[1])] = 0

    t = np.round(np.mean(x), r)

    mu1 = np.mean(x[x >= t])
    mu2 = np.mean(x[x < t])

    i = 1

    while True:
        cmu1 = np.mean(x[x >= t])
        cmu2 = np.mean(x[x < t])

        if (i > 1 and cmu1 == mu1 and cmu2 == mu2) or t > 1:
            log.info('Optimal threshold t = %s' % t)
            if t > cap:
                t = np.percentile(x, 90)
                log.warning('Optimal threshold too high, setting threshold to 90th percentile t = %s' % t)

            return t
        else:
            mu1 = cmu1
            mu2 = cmu2
            t = np.round((mu1 + mu2) / 2, r)
            log.debug('Iteration %s, threshold t = %s' % (i - 1, t))

        i += 1


def threshold_image(im, rows, scale_to_h=1000):
    im_small = rescale(im, scale_to_h / im.shape[0], mode='reflect', multichannel=False, anti_aliasing=True)

    si = np.round((im_small.shape[0] / rows) * 1.5)

    op = opening(im_small, square(round_odd(si)))
    op = resize(op, im.shape, mode='reflect')
    im = np.clip(im - op, 0, 1)

    # Find threshold
    thresh = _find_optimal_threshold(
        rescale(im, scale_to_h / im.shape[0], mode='reflect', multichannel=False, anti_aliasing=True))

    return (im >= thresh).astype(np.uint8)


def _split_half(vec):
    t = int(np.ceil(vec.shape[0] / 2))  # middle
    return vec[:t - 1], vec[t + 1:]


def _rle(threshold):
    """
    Fast run length encoding function. Adapted from https://stackoverflow.com/a/1066838
    This is a wrapper that can be used as

    :param threshold: threshold value used in the wrapped function
    :type threshold: ndarray

    :return: RLE function
    :rtype: function
    """

    def _inner(bits):
        # make sure all runs of ones are well-bounded
        bounded = np.hstack(([0], bits, [0]))
        # get 1 at run starts and -1 at run ends
        difs = np.diff(bounded)
        run_starts, = np.where(difs > 0)
        run_ends, = np.where(difs < 0)
        return np.any((run_ends - run_starts) > threshold)

    return _inner


def remove_rle(im, p=0.2, axis=0):
    c = p * im.shape[axis]

    z = np.apply_along_axis(_rle(c), axis, im)
    x = im.sum(axis=axis)

    zhl, zhr = _split_half(z)

    left, right = np.where(zhl)[0].max(), np.where(zhr)[0].min() + zhl.shape[0]

    if zhl.any():
        x[:left] = 0
    if zhr.any():
        x[right:] = 0

    return x, left, right


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def colony_peaks(x, n):
    # Smooth the signal
    window = signal.windows.general_gaussian(51, p=1.5, sig=20)

    filtered = signal.fftconvolve(window, x)
    filtered = (np.average(x) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -25)

    # Find local maximas
    peaks = signal.argrelmax(filtered, order=30)[0]

    peak_distances = np.abs(peaks[:-1] - peaks[1:])
    dist_dev = np.std(_rolling_window(peak_distances, n - 1), 1)
    grid_start = np.argmin(dist_dev)

    return np.median(peak_distances), peaks[grid_start:grid_start + n]


def find_bounds(spot_axis):
    half_axis = int(spot_axis.shape[0] / 2)

    left = -np.argmin(spot_axis[:half_axis][::-1]) - 1
    right = np.argmin(spot_axis[half_axis:])

    return left, right


def circularity(spot, area):
    if not area:
        return 0

    perim = perimeter(spot)

    if perim != 0:
        return (4 * np.pi * area) / (perim * perim)
    return 0
