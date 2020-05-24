import warnings


def simplified_format(
        message, category, filename, lineno, line=None):
    to_print = '{:s}:{:d}: {:s}: {:s}\n'.format(
        filename, lineno, category.__name__, str(message)
    )
    return to_print

warnings.formatwarning = simplified_format
