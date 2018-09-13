import warnings
from inspect import currentframe, getframeinfo

def warn_message_only(message, category=UserWarning):
    frameinfo = getframeinfo(currentframe())
    warnings.showwarning(
        message, category, frameinfo.filename, frameinfo.lineno,
        file=None, line=''
    )  # line='' supresses printing the line from codes.