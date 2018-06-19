DEFAULT_FORMAT = 1536

FORMATS = {
    1536: (32, 48),
    768: (32, 48),
    384: (16, 24),
    96: (8, 12)
}


class GitterException(Exception):
    pass
