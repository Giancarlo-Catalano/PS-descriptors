#!/usr/bin/env python3
import logging
import sys
import traceback
import warnings

from LCS.test_LCS import test_if_library_works, test_manually, test_custom_problem
from LightweightLocalPSMiner.SolutionRowCacher import test_solution_row_cacher


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = warn_with_traceback

    # test_LCS(optimisation_problem= MultiPlexerProblem(),
    #          rule_population_size=100,
    #          solution_count=1000,
    #          training_repeats=30)

    # test_multivariate_importance()

    #test_local_linkage()

    #test_custom_problem()

    test_solution_row_cacher()









