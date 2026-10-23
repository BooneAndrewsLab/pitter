import logging

import numpy as np
from scipy import signal
from skimage.measure import perimeter
from skimage.morphology import binary_opening, disk, opening, remove_small_objects, square
from skimage.transform import radon, rescale, rotate, resize, downscale_local_mean

from .common import GitterException

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
    im = rescale(im, scale_to_h / im.shape[0], mode='reflect', anti_aliasing=True)
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
        above, below = x[x >= t], x[x < t]
        # An empty side makes its mean NaN, and NaN never compares equal, so the loop
        # would never end. Rounding can also leave t alternating between two values.
        if not above.size or not below.size:
            raise GitterException('The image has no contrast to threshold colonies from. '
                                  'Check that it is a plate photograph.')
        if i > 100:
            raise GitterException('Could not find a threshold for this image.')

        cmu1 = np.mean(above)
        cmu2 = np.mean(below)

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
    scale = int(np.floor(1. / (scale_to_h / im.shape[0])))
    im_small = downscale_local_mean(im, (scale, scale))  # downscaling is faster than resizing

    si = np.round((im_small.shape[0] / rows) * 1.5)

    op = opening(im_small, square(round_odd(si)))
    im_small = np.clip(im_small - op, 0, 1)

    op = resize(op, im.shape, mode='reflect')

    im = np.clip(im - op, 0, 1)

    # Find threshold
    thresh = _find_optimal_threshold(im_small)

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


def remove_rle(im, p=0.2, axis=0, override_boundaries=None):
    c = p * im.shape[axis]

    z = np.apply_along_axis(_rle(c), axis, im)
    x = im.sum(axis=axis)

    if override_boundaries:
        left, right = override_boundaries
        x[:left] = 0
        x[right:] = 0
    else:
        zhl, zhr = _split_half(z)

        # A half without a long run has no plate edge in the picture (the scan is cropped
        # close to the plate, or the rim thresholded in pieces), so that side stays as is.
        # _split_half drops the two middle elements, hence the right half's offset.
        left = np.where(zhl)[0].max() if zhl.any() else 0
        right = np.where(zhr)[0].min() + x.shape[0] - zhr.shape[0] if zhr.any() else x.shape[0]

        x[:left] = 0
        x[right:] = 0

    return x, left, right


def _rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


#: Peaks lower than this fraction of the median peak height are not colony rows/columns.
#: The agar rim and plate edge leave a smoothed peak at roughly 0.4x of a real row, and
#: they are as evenly spaced as a real row, so spacing alone cannot tell them apart.
MIN_PEAK_HEIGHT = 0.5

#: Windows whose spacing deviation is within this many pixels of the best one are counted
#: as equally good placements. Peak positions are whole pixels, so a window that differs
#: by one pixel of spread is indistinguishable from the best; a pitch is ~100px, so this
#: still separates a regular grid from a clearly irregular one.
AMBIGUITY_TOLERANCE_PX = 1.0

#: Line spacing, in pixels, that the smoothing window and the minimum peak distance in
#: colony_peaks are tuned for.
PEAK_TUNED_PITCH = 100.

#: Below this estimated spacing the tuned settings are scaled down by spacing / tuned
#: spacing, or neighbouring lines blur into one peak: a small scan with 1536 colonies
#: across 1500px has them ~26px apart. Camera images of 1536 plates estimate at ~70-100px
#: and grid well with the tuned settings, so they are left exactly as they were.
PEAK_SCALE_BELOW_PITCH = 60.


def _expected_pitch(x, n):
    """
    Upper bound on the distance between colony lines: the stretch of the projection that
    has any signal, split into n. It includes the margins around the grid, so it errs
    towards the tuned peak finding settings rather than away from them.
    """
    on = np.flatnonzero(x > 0)
    span = on[-1] - on[0] + 1 if len(on) else len(x)
    return span / float(n)


def colony_peaks(x, n, border_to_zero=False, axis_name='lines', info=None):
    """
    Find the n colony lines in a row or column projection.

    Peaks are the local maxima of the smoothed projection; the n consecutive peaks with
    the most even spacing are the grid. Peaks much lower than the typical one (the plate
    rim) are discarded first.

    :param x: Projection of the thresholded plate onto one axis
    :param n: Number of rows/columns the plate format has
    :param border_to_zero: Zero the signal below its mean before looking for peaks
    :param axis_name: Name used in error messages
    :param info: Optional dict that is filled with 'ambiguous' (more than one window is
        about as regular as the best) and 'candidates' (how many such windows)
    :type x: ndarray
    :type n: int
    :type info: dict

    :return: Median peak distance and the n peak positions
    :rtype: tuple
    """
    pitch = _expected_pitch(x, n)
    scale = pitch / PEAK_TUNED_PITCH if pitch < PEAK_SCALE_BELOW_PITCH else 1.
    length = round_odd(51 * scale)

    # Smooth the signal
    window = signal.windows.general_gaussian(length, p=1.5, sig=20 * scale)

    filtered = signal.fftconvolve(window, x)
    filtered = (np.average(x) / np.average(filtered)) * filtered
    filtered = np.roll(filtered, -(length // 2))

    if border_to_zero:
        filtered[filtered < np.mean(filtered)] = 0

    # Find local maximas
    peaks = signal.argrelmax(filtered, order=max(1, round(30 * scale)))[0]

    # Peaks outside the plate were zeroed by remove_rle, so they must not drag the median
    # down. The FFT leaves ~1e-12 of rounding noise where the signal is zero, hence a
    # relative threshold rather than > 0.
    heights = filtered[peaks]
    real = heights > 1e-6 * heights.max() if len(heights) else heights > 0
    if np.any(real):
        peaks = peaks[heights >= MIN_PEAK_HEIGHT * np.median(heights[real])]

    # Fewer peaks than the plate needs means the plate format is wrong (a 384 plate gridded
    # as 1536, say); left alone, the rolling window below fails with a numpy shape error.
    if len(peaks) < n:
        raise GitterException(f'Found only {len(peaks)} {axis_name} of colonies, but this plate format '
                              f'needs {n}. Check that the plate format matches the plate.')

    peak_distances = np.abs(peaks[:-1] - peaks[1:])
    dist_dev = np.std(_rolling_window(peak_distances, n - 1), 1)
    grid_start = np.argmin(dist_dev)

    if info is not None:
        candidates = int(np.sum(dist_dev <= dist_dev[grid_start] + AMBIGUITY_TOLERANCE_PX))
        info['candidates'] = candidates
        info['ambiguous'] = candidates > 1

    return np.median(peak_distances), peaks[grid_start:grid_start + n]


def remove_noise(im, window):
    """
    Drop speckle from a thresholded image so it does not count towards colony sizes.

    Smudges, handwriting and plate edges threshold into speckle that fills the gaps
    between colonies. find_bounds then finds no gap, the colony's box grows towards its
    neighbours, and every speck inside it is counted as colony. An opening cuts the thin
    bridges between speckle and colonies, and the loose specks left are removed. Both
    sizes scale with the grid spacing, so the result does not depend on image resolution.

    :param im: thresholded image
    :param window: distance between neighbouring colonies, in pixels
    :type im: ndarray
    :type window: float

    :return: Thresholded image without the noise
    :rtype: ndarray
    """
    im = binary_opening(im.astype(bool), disk(max(1, int(round(window / 20)))))
    im = remove_small_objects(im, max_size=int(round((window / 8) ** 2)))
    return im.astype(np.uint8)


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
