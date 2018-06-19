import click

from ..core import Gitter


@click.command()
@click.argument("image")
@click.option('--plate-format', default=1536, help='Plate format as colony count.')
@click.option('--remove-noise', default=False, help='Try removing nose from the image.')
@click.option('--auto-rotate', default=False, help='Fix image rotation.')
@click.option('--inverse', default=False, help='Work on inverse image.')
@click.option('--contrast', default=None, help='Adjust image contrast.')
@click.option('--fast', default=0, help='Resize image before working on it. This speeds up the processing')
@click.option('--save-grid', default=False, help='Save gridded image. Helps with debugging.')
@click.option('--save-dat', default=True, help='Save dat file.')
def main(image, **kwargs):
    Gitter.auto_process(image, **kwargs)


if __name__ == '__main__':
    main()
