from ..core import Gitter


def main():
    import argparse
    import os
    from glob import glob

    parser = argparse.ArgumentParser(
        description='Process SGA plate(s) and measure colony sizes.',
        epilog="")
    parser.add_argument('image', nargs='+', help='Image(s) or folder(s) to process.')
    parser.add_argument('-f', '--plate-format', default=1536, help='Plate format as colony count. Default is 1536.')
    parser.add_argument('-n', '--remove-noise', action='store_true', help='Try removing noise from the image.')
    parser.add_argument('-r', '--auto-rotate', action='store_true', help='Fix image rotation.')
    parser.add_argument('-i', '--inverse', action='store_true', help='Work on inverse image.')
    parser.add_argument('-c', '--contrast', type=float, default=None, help='Adjust image contrast.')
    parser.add_argument('-s', '--rescale', default=0,
                        help='Resize image before working on it. This speeds up the processing.')
    parser.add_argument('-g', '--save-grid', action='store_true', help='Save gridded image. Helps with debugging.')
    parser.add_argument('-d', '--skip-dat', action='store_true', help='Save dat file.')
    parser.add_argument('-e', '--ignore-errors', action='store_true', help='Ignore processing errors, continue processing next image.')
    parser.add_argument('-R', '--resume-processing', action='store_true', help='Resume processing, skip images with existing dat files.')
    parser.add_argument('-C', '--colony-compat', action='store_true',
                        help='Enable colony imager compatibility mode (header).')

    args = vars(parser.parse_args())
    args['save_dat'] = not args.pop('skip_dat')

    images = []

    for im in args.pop('image'):
        if os.path.isdir(im):
            for fim in glob(os.path.join(im, '*JPG')):
                images.append(fim)
        elif os.path.isfile(im):
            images.append(im)
    
    failed = []
    resume = args.pop('resume_processing')
    ignore_errors = args.pop('ignore_errors')

    print("Processing %d images" % len(images))
    for imidx, im in enumerate(images):
        if resume and os.path.exists(os.path.splitext(im)[0] + '.dat'):
            print("Skipping %d/%d: %s" % (imidx + 1, len(images), im))
            continue

        print("Image %d/%d: %s" % (imidx + 1, len(images), im))
        try:
            Gitter.auto_process(im, **args)
        except Exception as ex:
            if ignore_errors:
                failed.append(im)
            else:
                raise ex

    if ignore_errors and failed:
        print("Failed to process these images:")
        for im in failed:
            print("\t%s" % im)


if __name__ == '__main__':
    main()
