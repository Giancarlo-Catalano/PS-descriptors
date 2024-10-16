import itertools
import random

import numpy as np
from tqdm import tqdm

import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
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

        def get_jaccard_distance(ps_a: PS, ps_b: PS) -> float:
            fixed_a = ps_a.values != STAR
            fixed_b = ps_b.values != STAR
            intersection = np.sum(fixed_a & fixed_b)
            union = np.sum(fixed_a | fixed_b)

            return float(intersection / union)

        hamming_distances = [PS.get_hamming_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        jaccard_distances = [PS.get_jaccard_distance(a, b)
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
        hamming_distance = main_solution.get_hamming_distance(background_solution)

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
                                   background_solutions: list[FullSolution],
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


        print("Additionally, here is the ps in google sheets form")
        assert(isinstance(self.optimisation_problem, EfficientBTProblem))
        self.optimisation_problem.print_ps_for_google_sheets(pss[0])



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



    def get_saturday_score_of_solution(self, fs: FullSolution) -> float:
        assert (isinstance(self.optimisation_problem, EfficientBTProblem))
        fitness_breakdown = self.optimisation_problem.breakdown_of_fitness_function(fs)
        return fitness_breakdown["by_weekday"]["Saturday"]

    def get_solutions_with_better_saturdays(self, main_solution: FullSolution) -> list[FullSolution]:
        # they are also sorted by similarity
        assert(isinstance(self.optimisation_problem, EfficientBTProblem))

        solutions_and_satfits = [(solution, self.get_saturday_score_of_solution(solution))
                                  for solution in self.pRef.get_evaluated_FSs()]
        own_satfit = self.get_saturday_score_of_solution(main_solution)

        eligible_solutions = [solution
                                 for solution, satfit in solutions_and_satfits
                                 if satfit < own_satfit]  # also removes main_solution



        eligible_solutions.sort(key=lambda x: x.get_hamming_distance(main_solution))

        return eligible_solutions




    def get_explanation_to_improve_saturday(self):
        main_solution = self.pRef.get_best_solution()
        eligible_saturday_improvements = self.get_solutions_with_better_saturdays(main_solution)
        if len(eligible_saturday_improvements) == 0:
            print("Seems that the solution already has the best saturdays...")
        background_solution = eligible_saturday_improvements[0]
        """ It would be interesting to plot the background solutions with
                x_axis = satfit
                y_axis = hamming_distance(main_solution)"""



        def prune_tradeoff(hamming_sat_data: list[(int, float)]) -> list[(int, float)]:
            all_hamming_distances = set(hamming for hamming, satfit in hamming_sat_data)
            best_for_hamming_distance_dict = dict()
            for hamming_distance_category in all_hamming_distances:
                best_for_hamming_distance_dict[hamming_distance_category] = min(satfit for hamming, satfit in hamming_sat_data if hamming == hamming_distance_category)

            return list(best_for_hamming_distance_dict.items())

        interesting_data = [
            (alternative.get_hamming_distance(main_solution), self.get_saturday_score_of_solution(alternative))
            for alternative in eligible_saturday_improvements]

        pruned_interesting_data = prune_tradeoff(interesting_data)

        main_solution_satfit = self.get_saturday_score_of_solution(main_solution)
        background_satfit = self.get_saturday_score_of_solution(background_solution)


        print(f"You are intending to improve the Saturday score of the known optima, which has satfit = {main_solution_satfit}")
        print(f"The background solution that has better satfit and is the closest "
              f"has hamming distance = {background_solution.get_hamming_distance(main_solution)},"
              f"and satfit = {background_satfit}")

        descriptors_manager = self.get_temporary_descriptors_manager()
        self.produce_explanation_sample(main_solution=main_solution,
                                        background_solutions=[background_solution],
                                        descriptors_manager=descriptors_manager)











