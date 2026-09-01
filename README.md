# pitter

Colony quantification from plate images: fit a grid to a photograph of a pinned agar
plate, and measure every colony on it.

A Python reimplementation of [`gitter`](https://github.com/omarwagih/gitter), Omar Wagih's
R package, and a replacement rather than a port — it adds gridding templates, liquid
timecourse assays and illumination correction, none of which the R has.

**It imports as `gitter`, not `pitter`.** The repository is named for the rewrite; the
distribution kept the name the imports use, because the code it replaces is called
`gitter` everywhere it is referenced.

## Install

```bash
mamba env create -f environment.yml
pip install -e .
```

Dependencies are pandas, numpy, scipy, scikit-image and matplotlib. Nothing is pinned.

## Using it

As a command:

```bash
gitter plate.jpg                 # writes plate.dat beside the image
gitter -f 384 -g images/         # a 384 plate, keeping the gridded overlay
gitter -w -e -R screen/          # recurse, skip failures, resume an interrupted run
```

| | |
|---|---|
| `-f, --plate-format` | colony count: 1536 (default), 768, 384, 96 |
| `-i, --inverse` | colonies are darker than the plate behind them |
| `-r, --auto-rotate` | correct a plate photographed askew |
| `-c, --contrast` | adjust contrast before thresholding |
| `-s, --rescale` | resize to this width first (1500–4000px); faster |
| `-g, --save-grid` | write the gridded overlay — see below |
| `--grid-on-thresholded` | draw that overlay on the mask rather than the plate |
| `-d, --skip-dat` | do not write the `.dat` |
| `-C, --colony-compat` | write the Colony Imager header instead of this one |
| `-t, --template-plate` | grid every plate from this one's fit |
| `-l, --detect-template` | use each plate's last timepoint as its own template |
| `-T, --use-template-locations` | take the template's colony coordinates, not just the plate's |
| `-L, --liquid-assay` | timecourse liquid assay: measure opacity, not area |
| `--local-illumination` | per-colony background for liquid assays. Slow, and a work in progress |
| `-z, --zero-border` | suppress "ghost" peaks at the plate border |
| `-w, --recurse`, `-e, --ignore-errors`, `-R, --resume-processing` | batch behaviour |

As a library:

```python
from gitter.core import Gitter

g = Gitter.auto_process('plate.jpg', plate_format=1536, save_dat='plate.dat')
g.data          # row, col, size, circularity
```

`auto_process` is `load_image()`, `grid()`, `quantify()`, `save()` in order; call them
yourself when you want the intermediates.

## What it writes

**`<image>.dat`** — the measurements, tab separated, sorted by row then column:

```
#Dat-format-version: 1
#Plate-boundaries: 251,203,3645,2369
#Grid-boundaries: 389,263,3510,2322
#Window: 66
row	col	size	circularity
1	1	3770.0	0.5076335590551022
```

The `#` lines record where the plate and the fitted grid were found, and are carried
through as provenance by anything that reads this. `--colony-compat` writes the Colony
Imager's own header instead, which is 13 lines and no column header; note that it
hardcodes `48 32 1536`, so it is only correct for a 1536 plate.

**`<image>_gridded.jpg`** — the fitted grid drawn over the plate, with a red box per
colony marking the bounds that were measured. Not decoration: it is the only way to see
that a plate was measured where you think it was, and a screen quantified without ever
looking at one is a screen nobody has checked.

## Performance, and why the code looks the way it does

A plate took **14.84s**. It now takes **2.20s**, and the `.dat` output is byte-identical
throughout — every change below is to how the work is arranged, not what is computed.
Measured on a 3888×2592 plate, 1536 colonies:

```
                original    now
load_image        0.84s     0.89s   imread + threshold_image
grid              0.26s     0.28s
quantify          2.96s     0.60s
save             10.78s     0.44s
TOTAL            14.84s     2.20s
```

Nearly all of it was overhead attached to **drawing the picture** rather than measuring
the plate — and not only inside `save()`. Two thirds of `quantify` was bookkeeping for a
JPEG that had not been written yet.

### Drawing the overlay: 10.78s → 0.44s

The `.dat` half of `save()` is 0.01s. All of the rest was the overlay, from two
independent causes:

- **6144 matplotlib artists.** `vlines`, `hlines` and two `plot` calls per colony, in a
  Python loop. **4.63s.** Now one `LineCollection` holding a closed path per colony:
  **0.06s**.
- **An 8000×4000 canvas.** `set_size_inches((40, 20))` at `dpi=200`, for a 3888×2592
  photograph — so the image was upscaled twofold and surrounded by white margin and axis
  ticks labelling pixel coordinates. **5.77s.** Now the figure is sized to the image and
  the axis is off: **1.42s** at full size, **0.80s** at the half resolution it writes.

Then a third, once those two stopped hiding it: matplotlib was **resampling the full-size
array down to the half-size canvas**, 0.84s of it. Decimating with a stride slice before
handing the array over, and using `extent` to keep the box coordinates in the full image's
pixel space, gives the same nearest-neighbour picture at the same file size for **0.44s**.

It also no longer goes through `pyplot`. `Figure` and `FigureCanvasAgg` are used directly,
which means there is no global figure to remember to close (the old code called `clf()`,
not `close()`), no shared state to make `save()` unsafe off the main thread, and **no
backend for a headless process to guess at** — importing this package no longer imports
`pyplot` at all, so a worker with no display needs no `matplotlib.use('Agg')`.

Result: **10.78s → 0.44s** to save, and **2806KB → 715KB** on disk.

### Measuring: 3.36s → 0.60s

`quantify_solid` collects `size` and `circularity` into plain numpy buffers and assigns
them to the frame once, at the end. The block that records the colony bounds for the
overlay did not: it wrote **eight `.loc[idx, ...]` scalars per colony**, which is 12288
pandas setitem calls and, in a profile of the whole run, 5.1s of cumulative time — the
largest single entry, larger than measuring every colony on the plate.

It now uses the same buffers, three lines above it, and assigns once. Nothing else
changed; `save_grid=False` was already 0.61s and is unaffected.

**A plate is now ~2.2s, and what remains is real work** — 0.89s decoding a 4MP JPEG and
thresholding it, 0.60s measuring 1536 colonies (over half of that is scikit-image's
`perimeter`, inside `circularity`), 0.44s writing the picture. The next worthwhile
speed-up is not per-plate: images are independent and nothing here processes them in
parallel, so a 40 plate screen is 88s serially against roughly 11s across eight cores.
`--template-plate` and `--detect-template` constrain the ordering, so a pool would have to
grid the templates first.

### The defaults, and what they cost

`GRID_SCALE`, `GRID_QUALITY` and `GRID_DPI` at the top of `core.py`. The overlay is
written at **half resolution, JPEG quality 85, over the plate photograph**. Measured
alternatives, same plate:

| | size |
|---|---|
| source photograph, for reference | 453KB |
| thresholded mask, full size, q85 *(what it used to write, minus the margins)* | 2134KB |
| thresholded mask, half resolution | 852KB |
| plate photograph, full size, q85 | 1425KB |
| **plate photograph, half resolution, q85** *(the default)* | **715KB** |

Half resolution still leaves roughly 40 pixels per colony on a 1536 plate, which is far
more than enough to see a grid that has slipped a row. At full size a 40 plate screen is
57MB of debugging aid.

Drawing over the **photograph** rather than the thresholded mask is the other half of it,
and it is not only smaller: the mask cannot show you a grid that fitted the mask
correctly while the mask itself was wrong. `--grid-on-thresholded` puts it back when what
you are debugging is the thresholding.

## Known issues

- **`--save-grid` with `--liquid-assay` will raise.** `save()` reads the `cl`/`cr`/`rl`/`rr`
  and `newx`/`newy` columns, and only `quantify_solid` writes them — `quantify_liquid` sets
  `size` and `circularity` and nothing else. Found by reading, not reproduced; there is no
  liquid timecourse data here to try it on.
- **scikit-image 0.27 will break `threshold_image`.** `gitter/utils.py` imports
  `morphology.square`, deprecated since 0.25 and removed in 0.27. The replacement is
  `footprint_rectangle`. Currently a `FutureWarning` on every image.
- **The blue marker is dead.** The overlay draws a blue dot at `(x, y)` and a yellow one at
  `(newx, newy)`, meant to show a colony that was recentred. `quantify_solid` assigns the
  same value to both, so they coincide on every colony of every plate — measured 0 of 1536
  differing. Either the recentring should record where it started, or one marker should go.
- **`core.py` has commented-out `plt` debugging lines** in `grid()` and `quantify_solid()`.
  They no longer have an import to go with them; uncommenting one now needs
  `from matplotlib import pyplot as plt` added back.
- **There are no tests.** `test/` is empty and `test.py` is a one-line CLI driver. The
  rewrites above were verified by comparing `.dat` output before and after, which catches
  a regression in the measurement but not one in the picture.
- **One unexplained SIGSEGV, 2026-09-01.** Seen once while reworking `quantify_solid`, on
  the first run after the edit, with stderr discarded — so nothing was captured beyond the
  core file. It did not recur in 34 subsequent runs of the same workload on the same
  image. Native code from numpy, scipy, scikit-image and matplotlib is all in play and it
  is not attributed to anything; recorded here so that a second occurrence is known to be
  a second rather than a first.

## Who uses it

[`sga_tools_web`](https://github.com/BooneAndrewsLab/sga_tools_web), the SGAtools web front
end, as the image analysis half — the `.dat` it writes is the input to
[`sga_score`](https://github.com/BooneAndrewsLab/sga_score), which normalizes and scores it.
That join is why `sga_score.read.read_dat` finds where a file's data starts rather than
skipping a fixed number of header lines: this package writes a column header and the
Colony Imager format does not.
