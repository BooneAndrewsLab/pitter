import logging

from gitter.scripts import gitter

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

gitter.main()
