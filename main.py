#!/usr/bin/env python3
import logging
import sys
import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = warn_with_traceback

    print("Nothing to see here...")











