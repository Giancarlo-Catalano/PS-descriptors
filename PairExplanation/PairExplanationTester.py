import itertools

import numpy as np
from tqdm import tqdm

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Explanation.PRefManager import PRefManager
from LCS import PSEvaluator
from LCS.ConstrainedPSSearch.SolutionDifferencePSSearch import local_constrained_ps_search
from LCS.PSEvaluator import GeneralPSEvaluator
from utils import announce


class PairExplanationTester:
    optimisation_problem: BenchmarkProblem
    ps_search_budget: int
    ps_search_population_size: int

    fs_evaluator: FSEvaluator
    ps_evaluator: GeneralPSEvaluator

    pRef_creation_method: str
    pRef_size: int

    pRef: PRef

    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 ps_search_budget: int,
                 ps_search_population: int,
                 pRef_creation_method: str = "uniform GA",
                 pRef_size: int = 10000,
                 verbose: bool = False):
        self.verbose = verbose

        self.optimisation_problem = optimisation_problem
        self.ps_search_budget = ps_search_budget
        self.ps_search_population_size = ps_search_population
        self.pRef_size = pRef_size

        self.pRef_creation_method = pRef_creation_method

        self.fs_evaluator = FSEvaluator(optimisation_problem.fitness_function)

        with announce(f"Creating the pRef of size {self.pRef_size}, method = {self.pRef_creation_method}",
                      self.verbose):
            self.pRef = self.generate_pRef()

        self.ps_evaluator = GeneralPSEvaluator(optimisation_problem=self.optimisation_problem, pRef=self.pRef)

    def generate_pRef(self) -> PRef:
        return PRefManager.generate_pRef(problem=self.optimisation_problem,
                                         which_algorithm=self.pRef_creation_method,
                                         sample_size=self.pRef_size)

    def get_consistency_of_pss(self, pss: list[PS]) -> dict:
        def get_hamming_distance(ps_a: PS, ps_b: PS) -> int:
            return np.sum(ps_a.values != ps_b.values)

        def get_jaccard_distance(ps_a: PS, ps_b: PS) -> float:
            fixed_a = ps_a.values != STAR
            fixed_b = ps_b.values != STAR
            intersection = np.sum(fixed_a & fixed_b)
            union = np.sum(fixed_a | fixed_b)

            return intersection / union

        hamming_distances = [get_hamming_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        jaccard_distances = [get_jaccard_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        return {"hamming_distances": hamming_distances,
                "jaccard_distances": jaccard_distances}

    def consistency_test_on_solution_pair(self,
                                          main_solution: FullSolution,
                                          background_solution: FullSolution,
                                          only_return_biggest: bool = False,
                                          only_return_least_dependent: bool = False,
                                          runs: int = 100,
                                          verbose: bool = False):
        if self.verbose:
            print(f"consistency_test_on_solution_pair({main_solution = }, "
                  f"{background_solution = }, "
                  f"{runs =}, "
                  f"{only_return_least_dependent = }, "
                  f"{only_return_biggest}")

        def single_test():
            return local_constrained_ps_search(to_explain=main_solution,
                                               background_solution=background_solution,
                                               population_size=self.ps_search_population_size,
                                               ps_evaluator=self.ps_evaluator,
                                               ps_budget=self.ps_search_budget,
                                               only_return_least_dependent=only_return_least_dependent,
                                               only_return_biggest=only_return_biggest,
                                               verbose=verbose)

        pss = []
        for run_index in tqdm(range(runs)):
            pss.extend(single_test())

        return self.get_consistency_of_pss(pss)

    def consistency_test_on_optima(self,
                                   runs: int,
                                   only_return_biggest=False,
                                   only_return_least_dependent=False) -> dict:
        optima = self.pRef.get_top_n_solutions(1)[0]
        closest_to_optima = PRefManager.get_most_similar_solution_to(pRef=self.pRef,
                                                                     solution=optima)  # excludes the solution itself

        return self.consistency_test_on_solution_pair(optima, closest_to_optima,
                                                      only_return_biggest=only_return_biggest,
                                                      only_return_least_dependent=only_return_least_dependent,
                                                      runs=runs)
