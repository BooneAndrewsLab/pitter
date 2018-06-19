import logging
import numpy as np
from scipy import signal
from skimage.transform import radon, rescale, rotate, resize
from skimage.morphology import opening, square
from skimage.measure import regionprops

log = logging.getLogger(__name__)


def set_contrast(im, contrast=10):
    c = (100.0 + contrast) / 100.0
    im = ((im - 0.5) * c) + 0.5
    return np.clip(im, 0, 1)


def round_odd(x):
    x = np.floor(x)
    return int(x + 1 - x % 2)


def round_even(x):
    return 2. * np.round(x / 2)


def find_rotation_angle(im, degree_inc=0.2, scale_to_h=500, angle_eps=50):
    im = rescale(im, scale_to_h / im.shape[0], mode='reflect')
    samples = (angle_eps * 2) / degree_inc

    theta = np.linspace(90 - angle_eps, 90 + angle_eps, int(samples), endpoint=False)
    sinogram = radon(im, theta=theta, circle=True)

    row_var = np.apply_along_axis(np.var, 0, sinogram)

    peaks, = signal.argrelmax(row_var, order=int(10 / degree_inc))
    peak = peaks[np.argmin(np.abs(peaks - (samples / 2)))]

    return ((samples / 2) - peak) * degree_inc


def autorotate_image(im, **kwargs):
    angle = find_rotation_angle(im, **kwargs)
    log.info('Rotate for %.2f' % (angle,))
    return rotate(im, angle)


def find_optimal_threshold(x, r=2, lim=(0, 0.4), cap=0.2):
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
                log.warn('Optimal threshold too high, setting threshold to 90th percentile t = %s' % t)

            return t
        else:
            mu1 = cmu1
            mu2 = cmu2
            t = np.round((mu1 + mu2) / 2, r)
            log.debug('Iteration %s, threshold t = %s' % (i - 1, t))

        i += 1


def threshold_image(im, rows, scale_to_h=1000):
    im_small = rescale(im, scale_to_h / im.shape[0], mode='reflect')

    si = np.round((im_small.shape[0] / rows) * 1.5)

    op = opening(im_small, square(round_odd(si)))
    op = resize(op, im.shape, mode='reflect')
    im = np.clip(im - op, 0, 1)

    # Find threshold
    thresh = find_optimal_threshold(rescale(im, scale_to_h / im.shape[0], mode='reflect'))

    return (im >= thresh).astype(np.uint8)


def split_half(vec):
    t = int(np.ceil(vec.shape[0] / 2))  # middle
    return vec[:t - 1], vec[t + 1:]


# Adapted from
# https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    """
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    """
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return np.array(run_lengths)


def remove_rle(im, p=0.2, axis=0):
    c = p * im.shape[axis]

    z = np.apply_along_axis(lambda xx: np.any(rle_encoding(xx)[1::2] > c), axis, im)
    x = im.sum(axis=axis)

    zhl, zhr = split_half(z)

    left, right = np.where(zhl)[0].max(), np.where(zhr)[0].min() + zhl.shape[0]

    if zhl.any():
        x[:left] = 0
    if zhr.any():
        x[right:] = 0

    return x, left, right


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def colony_peaks(x, n):
    # Smooth the signal
    window = signal.general_gaussian(51, p=1.5, sig=20)

    filtered = signal.fftconvolve(window, x)
    filtered = (np.average(x) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -25)

    # Find local maximas
    peaks = signal.argrelmax(filtered, order=30)[0]

    peak_distances = np.abs(peaks[:-1] - peaks[1:])
    dist_dev = np.std(rolling_window(peak_distances, n - 1), 1)
    grid_start = np.argmin(dist_dev)

    return np.median(peak_distances), peaks[grid_start:grid_start + n]


def find_bounds(spot_axis):
    half_axis = int(spot_axis.shape[0] / 2)

    left = -np.argmin(spot_axis[:half_axis][::-1]) - 1
    right = np.argmin(spot_axis[half_axis:])

    return left, right


def circularity(spot):
    r = regionprops(spot)
    if not r:
        return 0.

    r = r[0]
    return (4 * np.pi * r.area) / (r.perimeter * r.perimeter)
