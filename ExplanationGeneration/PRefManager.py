from math import ceil
from typing import Optional, Literal

import numpy as np
from scipy.stats import t

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PRef import PRef, plot_solutions_in_pRef
from FirstPaper.PS import PS
from FirstPaper.PSMetric.Classic3 import PSSearchMetricsEvaluator
from PSMiners.Mining import get_history_pRef
from utils import announce


class PRefManager:
    problem: BenchmarkProblem
    pRef_file: str

    cached_pRef: Optional[PRef]
    pRef_mean: Optional[float]
    evaluator: Optional[PSSearchMetricsEvaluator]

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 verbose: bool = False):
        self.problem = problem
        self.pRef_file = pRef_file
        self.cached_pRef = None
        self.evaluator = None
        self.pRef_mean = None
        self.verbose = verbose

    @staticmethod
    def generate_pRef(problem,
                      sample_size: int,
                      which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"],
                      force_include: Optional[list[FullSolution]] = None,
                      verbose: bool = False) -> PRef:

        methods = which_algorithm.split()
        sample_size_for_each = ceil(sample_size / len(methods))

        def make_pRef_with_method(method: str) -> PRef:
            return get_history_pRef(benchmark_problem=problem,
                                 which_algorithm=method,
                                 sample_size=sample_size_for_each,
                                 verbose=verbose)

        pRefs = [make_pRef_with_method(method) for method in methods]

        if force_include is not None and len(force_include) > 0:
            forced_pRef = PRef.from_full_solutions(force_include,
                                                   fitness_values=[problem.fitness_function(fs) for fs in force_include],
                                                   search_space=problem.search_space)
            pRefs.append(forced_pRef)

        return PRef.concat(pRefs)

    def instantiate_evaluator(self):
        self.evaluator = PSSearchMetricsEvaluator(self.cached_pRef)


    def instantiate_mean(self):
        self.pRef_mean = np.average(self.cached_pRef.fitness_array)

    def generate_pRef_file(self, sample_size: int,
                           which_algorithm,
                           force_include: Optional[list[FullSolution]] = None):
        """ options for which_algorithm are "uniform", "GA", "SA", "GA_best", "SA_best",
        you can use multiple by space-separating them, eg "uniform SA" """

        self.cached_pRef = PRefManager.generate_pRef(self.problem,
                                                     sample_size,
                                                     which_algorithm,
                                                     force_include=force_include,
                                                     verbose=self.verbose)
        plot_solutions_in_pRef(self.cached_pRef)
        #self.instantiate_evaluator()
        self.instantiate_mean()

        with announce(f"Writing the pRef to {self.pRef_file}", self.verbose):
            self.cached_pRef.save(file=self.pRef_file)




    @property
    def pRef(self) -> PRef:
        if self.cached_pRef is None:
            self.cached_pRef = PRef.load(self.pRef_file)
            #self.instantiate_evaluator()
            self.instantiate_mean()
        return self.cached_pRef


    def t_test_for_mean_with_ps(self, ps: PS) -> (float, float):
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return -1, -1

        t_score = (sample_mean - self.pRef_mean) / (sample_stdev / np.sqrt(n))
        p_value = 1 - t.cdf(abs(t_score), df=n - 1)
        return p_value, sample_mean

    def get_average_when_present_and_absent(self, ps: PS) -> (float, float):
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        observations, not_observations = self.pRef.fitnesses_of_observations_and_complement(ps)
        return np.average(observations), np.average(not_observations)


    def get_atomicity_contributions(self, ps: PS) -> np.ndarray:
        return self.evaluator.get_atomicity_contributions(ps, normalised=True)




