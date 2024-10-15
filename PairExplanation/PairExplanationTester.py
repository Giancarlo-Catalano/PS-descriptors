import itertools
import random

import numpy as np
from tqdm import tqdm

import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import ForcefulMannWhitneyU
from Explanation.PRefManager import PRefManager
from LCS import PSEvaluator
from LCS.ConstrainedPSSearch.SolutionDifferencePSSearch import local_constrained_ps_search
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from LCS.PSEvaluator import GeneralPSEvaluator
from utils import announce, execution_timer


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

    preferred_culling_method: str

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 ps_search_budget: int,
                 ps_search_population: int,
                 pRef_creation_method: str = "uniform GA",
                 pRef_size: int = 10000,
                 preferred_culling_method: str = "biggest",
                 verbose: bool = False):
        self.verbose = verbose
        self.preferred_culling_method = preferred_culling_method

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


    def find_pss(self, main_solution: FullSolution, background_solution: FullSolution, culling_method: str) -> list[PS]:
        return local_constrained_ps_search(to_explain=main_solution,
                                    background_solution=background_solution,
                                    population_size=self.ps_search_population_size,
                                    ps_evaluator=self.ps_evaluator,
                                    ps_budget=self.ps_search_budget,
                                    culling_method=culling_method,
                                    verbose=self.verbose)

    def get_consistency_of_pss(self, pss: list[PS]) -> dict:
        def get_hamming_distance(ps_a: PS, ps_b: PS) -> int:
            return int(np.sum(ps_a.values != ps_b.values))

        def get_jaccard_distance(ps_a: PS, ps_b: PS) -> float:
            fixed_a = ps_a.values != STAR
            fixed_b = ps_b.values != STAR
            intersection = np.sum(fixed_a & fixed_b)
            union = np.sum(fixed_a | fixed_b)

            return float(intersection / union)

        hamming_distances = [get_hamming_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        jaccard_distances = [get_jaccard_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        return {"hamming_distances": hamming_distances,
                "jaccard_distances": jaccard_distances}

    def consistency_test_on_solution_pair(self,
                                          main_solution: FullSolution,
                                          background_solution: FullSolution,
                                          culling_method: str,
                                          runs: int = 100):
        if self.verbose:
            print(f"consistency_test_on_solution_pair({main_solution = }, "
                  f"{background_solution = }, "
                  f"{runs =}, "
                  f"{culling_method = }")

        pss = []
        with execution_timer() as time:
            for run_index in tqdm(range(runs)):
                pss.extend(self.find_pss(main_solution,
                                         background_solution,
                                         culling_method=self.preferred_culling_method))

        runtime = time.runtime

        results = self.get_consistency_of_pss(pss)
        results["total_runtime"] = runtime
        results["runs"] = runs
        results["sizes"] = [ps.fixed_count() for ps in pss]
        return results

    def consistency_test_on_optima(self,
                                   runs: int,
                                   culling_method: str) -> dict:
        optima = self.pRef.get_top_n_solutions(1)[0]
        closest_to_optima = PRefManager.get_most_similar_solution_to(pRef=self.pRef,
                                                                     solution=optima)  # excludes the solution itself

        return self.consistency_test_on_solution_pair(optima, closest_to_optima,
                                                      culling_method=culling_method,
                                                      runs=runs)



    def get_accuracy_of_explanations_on_pair(self,
                                              main_solution: EvaluatedFS,
                                              background_solution: EvaluatedFS,
                                              p_value_tester: ForcefulMannWhitneyU):

        with execution_timer() as timer:
            ps = self.find_pss(main_solution,
                               background_solution,
                               culling_method=self.preferred_culling_method)[0]

        beneficial_p_value, maleficial_p_value = p_value_tester.check_effect_of_ps(ps)

        situation = "expected_positive" if main_solution > background_solution else "expected_negative"
        hamming_distance = int(np.sum(main_solution.values != background_solution.values))

        return {"situation": situation,
                "greater_p_value": beneficial_p_value,
                "lower_p_value": maleficial_p_value,
                "main_fitness": main_solution.fitness,
                "background_fitness": background_solution.fitness,
                "hamming_distance": hamming_distance,
                "time": timer.runtime}

    def accuracy_test(self,
                      amount_of_samples: int):
        def pick_random_solution_pair() -> (EvaluatedFS, EvaluatedFS):
            main_solution = self.pRef.get_nth_solution(index = random.randrange(self.pRef.sample_size))
            background_solution = PRefManager.get_most_similar_solution_to(pRef = self.pRef, solution=main_solution)
            return main_solution, background_solution


        mwu_tester = ForcefulMannWhitneyU(sample_size=1000,
                                          search_space=self.optimisation_problem.search_space,
                                          fitness_evaluator=self.fs_evaluator)
        results = []
        for iteration in tqdm(range(amount_of_samples)):
            main_solution, background_solution = pick_random_solution_pair()
            datapoint = self.get_accuracy_of_explanations_on_pair(main_solution, background_solution, mwu_tester)
            results.append(datapoint)


        return results


    def produce_explanation_sample(self,
                                   main_solution: EvaluatedFS,
                                   background_solutions: list[EvaluatedFS],
                                   descriptors_manager: DescriptorsManager):

        pss = []
        for background_solution in tqdm(background_solutions):
            new_pss = self.find_pss(main_solution,
                                    background_solution,
                                    culling_method=self.preferred_culling_method)
            pss.extend(new_pss)

        print(f"For the solution \n\t{self.optimisation_problem.repr_fs(main_solution)}\n, and the {len(background_solutions)} background solutions:")
        for background_solution, pattern in zip(background_solutions, pss):
            description = descriptors_manager.get_descriptors_string(ps=pattern)
            print(f"background = \n{utils.indent(self.optimisation_problem.repr_fs(background_solution))}")
            print(f"pattern = \n{utils.indent(self.optimisation_problem.repr_ps(pattern))}")
            print(f"description = \n{utils.indent(description)}")
            print(f"\n")


    def get_background_solutions(self, main_solution: EvaluatedFS, background_solution_count: int) -> list[EvaluatedFS]:
        return PRefManager.get_most_similar_solutions_to(pRef = self.pRef,
                                                         solution = main_solution,
                                                         amount_to_return=background_solution_count)


    def get_temporary_descriptors_manager(self) -> DescriptorsManager:
        pRef_manager = PRefManager(problem = self.optimisation_problem,
                                   pRef_file = None,
                                   instantiate_own_evaluator=False,
                                   verbose=True)
        pRef_manager.set_pRef(self.pRef)

        descriptors_manager = DescriptorsManager(optimisation_problem=self.optimisation_problem,
                           control_pss_file=None,
                           control_descriptors_table_file=None,
                           control_samples_per_size_category=1000,
                           pRef_manager=pRef_manager,
                           verbose=True)

        descriptors_manager.start_from_scratch()
        return descriptors_manager

    def get_random_explanation(self):
        solution_to_explain = self.pRef.get_nth_solution(index = random.randrange(self.pRef.sample_size))
        background_solutions = self.get_background_solutions(main_solution=solution_to_explain,
                                                             background_solution_count=5)

        descriptors_manager = self.get_temporary_descriptors_manager()
        self.produce_explanation_sample(main_solution=solution_to_explain,
                                        background_solutions=background_solutions,
                                        descriptors_manager = descriptors_manager)






