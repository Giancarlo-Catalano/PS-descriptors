import random
from typing import Literal

from deap.tools import Logbook

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from PSMiners.Mining import get_history_pRef, load_pss, write_pss_to_file
from utils import announce


def generate_control_PSs(search_space: SearchSpace, reference_pss: list[PS]) -> list[PS]:
    def pick_size() -> int:
        to_mimic = random.choice(reference_pss)
        wanted_size = to_mimic.fixed_count()
        return wanted_size

    amount_to_generate = len(reference_pss)
    return [PS.random_with_fixed_size(search_space, pick_size())
            for _ in range(amount_to_generate)]



def generate_matching_control_pss(search_space: SearchSpace,
                                  ps_file: str,
                                  control_file_destination: str,
                                  verbose=False):

    with announce(f"Loading the partial solutions from {ps_file}"):
        pss = load_pss(ps_file)

    with announce(f"Generating the control PSs and writing them to {control_file_destination}"):
        control_pss = generate_control_PSs(search_space, pss)
        write_pss_to_file(control_pss, control_file_destination)