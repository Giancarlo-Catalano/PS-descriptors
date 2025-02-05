from typing import Optional, Literal

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, STAR
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.DEAP.DEAPPSMiner import DEAPPSMiner
from PSMiners.Mining import get_ps_miner, write_evaluated_pss_to_file, load_pss, write_pss_to_file
from PSMiners.PyMoo.SequentialCrowdingMiner import SequentialCrowdingMiner
from utils import announce


class MinedPSManager:
    problem: BenchmarkProblem
    mined_ps_file: str
    control_pss_file: str

    cached_pss: Optional[list[EvaluatedPS]]
    cached_control_pss: Optional[list[PS]]


    def __init__(self,
                 problem: BenchmarkProblem,
                 mined_ps_file: Optional[str] = None,
                 control_ps_file: Optional[str] = None,
                 verbose: bool = False):
        self.problem = problem
        self.mined_ps_file = mined_ps_file
        self.control_pss_file = control_ps_file
        self.cached_control_pss = None
        self.cached_pss = None
        self.verbose = verbose


    def mine_pss_old(self,
                 pRef: PRef,
                 ps_miner_method: Literal["classic", "NSGA", "NSGA_experimental_crowding", "SPEA2", "sequential"],
                 ps_budget: int) -> list[EvaluatedPS]:
        algorithm = get_ps_miner(pRef, which=ps_miner_method)

        with announce(f"Running {algorithm} on {pRef} with {ps_budget =}", self.verbose):
            budget_limit = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget)
            coverage_limit = TerminationCriteria.SearchSpaceIsCovered()
            termination_criterion = budget_limit #TerminationCriteria.UnionOfCriteria(budget_limit, coverage_limit)
            algorithm.run(termination_criterion, verbose=self.verbose)

        result_ps = algorithm.get_results(None)
        result_ps = AbstractPSMiner.without_duplicates(result_ps)
        result_ps = [ps for ps in result_ps if not ps.is_empty()]

        return result_ps




    def generate_ps_file_old(self,
                         pRef: PRef,
                         ps_miner_method: Literal["classic", "NSGA", "NSGA_experimental_crowding", "SPEA2", "sequential"],
                         ps_budget: int):

        with announce(f"Mining the partial solutions using {ps_miner_method} and budget = {ps_budget}"):
            self.cached_pss = self.mine_pss_old(pRef, ps_miner_method, ps_budget)



        with announce(f"Writing the PSs onto {self.mined_ps_file}", self.verbose):
            write_evaluated_pss_to_file(self.cached_pss, self.mined_ps_file)

    def mine_pss(self,
                 pRef: PRef,
                 population_size: int,
                 ps_budget_per_run: int,
                 ps_budget_in_total: int) -> list[EvaluatedPS]:
        algorithm = SequentialCrowdingMiner(pRef = pRef,
                                            budget_per_run=ps_budget_per_run,
                                            population_size_per_run=population_size,
                                            which_algorithm="NSGAII",
                                            use_experimental_crowding_operator=True)

        with announce(f"Running {algorithm} on {pRef} with {ps_budget_in_total =}", self.verbose):
            budget_limit = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget_in_total)
            termination_criterion = budget_limit #TerminationCriteria.UnionOfCriteria(budget_limit, coverage_limit)
            algorithm.run(termination_criterion, verbose=self.verbose)

        result_ps = algorithm.get_results(None)
        result_ps = AbstractPSMiner.without_duplicates(result_ps)
        result_ps = [ps for ps in result_ps if not ps.is_empty()]

        return result_ps


    def generate_ps_file(self,
                         pRef: PRef,
                         population_size: int,
                         ps_budget_per_run: int,
                         ps_budget_in_total: int):

        with announce(f"Mining the partial solutions"):
            self.cached_pss = self.mine_pss(pRef=pRef,
                                            population_size=population_size,
                                            ps_budget_in_total=ps_budget_in_total,
                                            ps_budget_per_run=ps_budget_per_run)



        with announce(f"Writing the PSs onto {self.mined_ps_file}", self.verbose):
            write_evaluated_pss_to_file(self.cached_pss, self.mined_ps_file)


    @property
    def pss(self) -> list[EvaluatedPS]:
        if self.cached_pss is None:
            self.cached_pss = load_pss(self.mined_ps_file)
        return self.cached_pss

    def generate_control_pss(self, samples_for_each_category: int = 1000) -> list[PS]:
        sizes = {ps.fixed_count() for ps in self.pss}
        result = []
        for size in sizes:
            samples_for_sizes_generator = (PS.random_with_fixed_size(self.problem.search_space, size)
                                            for _ in range(samples_for_each_category))
            result.extend(samples_for_sizes_generator)
        return result

    def generate_control_pss_file(self, samples_for_each_category: int = 1000):
        with announce(f"Generating the control pss, with {samples_for_each_category} samples for each category"):
            self.cached_control_pss = self.generate_control_pss(samples_for_each_category)

        with announce(f"Writing the control PSs onto the file {self.control_pss_file}"):
            write_pss_to_file(self.cached_pss, self.control_pss_file)


    @property
    def control_pss(self):
        if self.cached_control_pss is None:
            self.cached_control_pss = load_pss(self.control_pss_file)
        return self.cached_control_pss


    @classmethod
    def sort_by_atomicity(cls, pss: list[EvaluatedPS]) -> list[EvaluatedPS]:
        def get_atomicity(ps: EvaluatedPS) -> float:
            return ps.metric_scores[2]

        return sorted(pss, key=get_atomicity, reverse=False)



    def get_coverage_stats(self) -> np.ndarray:
        def ps_to_fixed_values_tally(ps: PS) -> np.ndarray:
            return ps.values != STAR

        return sum(ps_to_fixed_values_tally(ps) for ps in self.pss) / len(self.pss)




