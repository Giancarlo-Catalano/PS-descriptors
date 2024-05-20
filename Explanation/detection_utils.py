import random
from typing import Literal

from deap.tools import Logbook

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from PSMiners.Mining import get_history_pRef, load_pss, write_pss_to_file
from utils import announce


def generate_control_PSs(search_space: SearchSpace, reference_pss: list[PS], samples_for_each_category: int) -> list[PS]:

    sizes = set(ps.fixed_count() for ps in reference_pss)


    result = []
    for size in sizes:
        samples_for_sizes_generator = (PS.random_with_fixed_size(search_space, size)
                                        for _ in range(samples_for_each_category))
        result.extend(samples_for_sizes_generator)

    return result