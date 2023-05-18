

import sys


__version__ = "1.0.0rc12"

msg = "captionvqa is compatible with Python 3.6 and newer."


if sys.version_info < (3, 6):
    raise ImportError(msg)
