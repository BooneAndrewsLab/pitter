import numpy as np
import pytest

from gitter.common import GitterException
from gitter.core import Gitter, GitterOptions
from gitter.utils import colony_peaks, remove_rle


def projection(centres, heights, length=3000, sigma=15):
    x = np.zeros(length)
    pos = np.arange(length)
    for c, h in zip(centres, heights):
        x += h * np.exp(-((pos - c) / sigma) ** 2 / 2)
    return x


def test_weak_rim_peak_is_dropped():
    # 8 real rows plus a rim line one pitch after them: equally regular, but low.
    centres = 300 + 100 * np.arange(9)
    heights = [1000] * 8 + [400]
    info = {}
    _, peaks = colony_peaks(projection(centres, heights), 8, info=info)
    assert list(peaks) == list(centres[:8])
    assert not info['ambiguous']


def test_equal_peaks_are_ambiguous():
    centres = 300 + 100 * np.arange(9)
    info = {}
    colony_peaks(projection(centres, [1000] * 9), 8, info=info)
    assert info['ambiguous'] and info['candidates'] == 2


def test_too_few_peaks_after_filtering_raises():
    centres = 300 + 100 * np.arange(8)
    with pytest.raises(GitterException):
        colony_peaks(projection(centres, [1000] * 7 + [100]), 8)


def test_bounds_validation():
    with pytest.raises(GitterException):
        GitterOptions(bounds=(10, 10, 5, 50))
    with pytest.raises(GitterException):
        GitterOptions(bounds=(1, 2, 3))
    assert GitterOptions(bounds=(1, 2, 30, 40)).bounds == (1, 2, 30, 40)


def test_bounds_are_clipped_to_the_image():
    g = Gitter('x.jpg', bounds=(-5, 10, 500, 90))
    g.thresholded = np.zeros((100, 200))
    assert g._clipped_bounds() == ((0, 200), (10, 90))
    g = Gitter('x.jpg', bounds=(300, 10, 500, 90))
    g.thresholded = np.zeros((100, 200))
    with pytest.raises(GitterException):
        g._clipped_bounds()


def test_check_grid_flags_empty_edge():
    import pandas as p
    g = Gitter('x.jpg')
    g.window = 20
    g.thresholded = np.ones((400, 400))
    rows, cols = np.meshgrid(np.arange(1, 6), np.arange(1, 6), indexing='ij')
    g.data = p.DataFrame({'x': 50 + 40 * cols.ravel(), 'y': 50 + 40 * rows.ravel(),
                          'row': rows.ravel(), 'col': cols.ravel(),
                          'size': np.where(rows.ravel() == 5, 5, 100)})
    codes = {(w['code'], w['axis'], w['side']) for w in g.check_grid()}
    assert ('edge_empty', 'row', 'end') in codes
    assert not any(c[0] == 'edge_empty' and c[1] == 'col' for c in codes)


def test_small_scan_lines_are_not_merged():
    # A 1536 plate scanned at ~1500px wide: 48 columns only 26px apart.
    centres = 120 + 26 * np.arange(48)
    _, peaks = colony_peaks(projection(centres, [1000] * 48, length=1500, sigma=4), 48)
    assert np.all(np.abs(peaks - centres) <= 1)


def test_remove_rle_without_a_plate_edge():
    # Colonies but no long run (plate edge) in either half: nothing is clipped.
    im = np.zeros((100, 200), dtype=np.uint8)
    im[40:60, 20:180:10] = 1
    x, left, right = remove_rle(im, axis=0)
    assert (left, right) == (0, 200)
    assert x.sum() == im.sum()


def test_remove_rle_clips_at_both_edges():
    im = np.zeros((100, 201), dtype=np.uint8)
    im[:, 10] = 1    # left plate edge
    im[:, 190] = 1   # right plate edge
    im[40:60, 30:170:10] = 1
    x, left, right = remove_rle(im, axis=0)
    assert (left, right) == (10, 190)
    assert x[:10].sum() == 0 and x[190:].sum() == 0


def test_greyscale_file_loads_as_floats(tmp_path):
    # A single-channel scan: as_gray leaves it uint8 0-255 unless converted.
    from skimage.io import imsave
    path = tmp_path / 'grey.png'
    im = np.full((1008, 1457), 120, dtype=np.uint8)
    for y in range(100, 900, 60):
        for x in range(100, 1350, 60):
            im[y:y + 20, x:x + 20] = 170
    imsave(path, im)
    g = Gitter(str(path), plate_format=96)
    loaded = g.load_image()
    assert loaded.dtype.kind == 'f' and 0 <= loaded.min() and loaded.max() <= 1


def test_threshold_without_contrast_raises():
    # Every value outside the threshold range leaves one side empty; this used to loop
    # forever on a NaN mean.
    from gitter.utils import _find_optimal_threshold
    with pytest.raises(GitterException):
        _find_optimal_threshold(np.full((50, 50), 0.9))
