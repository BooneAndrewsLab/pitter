from ..core import Gitter


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Process SGA plate(s) and measure colony sizes.',
        epilog="")
    parser.add_argument('image', nargs='+', type=argparse.FileType('r'), help='Image(s) to process.')
    parser.add_argument('-f', '--plate-format', default=1536, help='Plate format as colony count. Default is 1536.')
    parser.add_argument('-n', '--remove-noise', action='store_true', help='Try removing noise from the image.')
    parser.add_argument('-r', '--auto-rotate', action='store_true', help='Fix image rotation.')
    parser.add_argument('-i', '--inverse', action='store_true', help='Work on inverse image.')
    parser.add_argument('-c', '--contrast', type=float, default=None, help='Adjust image contrast.')
    parser.add_argument('-s', '--rescale', default=0,
                        help='Resize image before working on it. This speeds up the processing.')
    parser.add_argument('-g', '--save-grid', action='store_true', help='Save gridded image. Helps with debugging.')
    parser.add_argument('-d', '--skip-dat', action='store_true', help='Save dat file.')
    parser.add_argument('-C', '--colony-compat', action='store_true',
                        help='Enable colony imager compatibility mode (header).')

    args = vars(parser.parse_args())
    args['save_dat'] = not args.pop('skip_dat')

    for im in args.pop('image'):
        Gitter.auto_process(im.name, **args)


if __name__ == '__main__':
    main()
