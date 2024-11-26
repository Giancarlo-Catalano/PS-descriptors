import itertools
import random
from typing import Optional

import numpy as np
from tqdm import tqdm

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima, effect_string
from Explanation.PRefManager import PRefManager
from ThirdPaper.SolutionDifferencePSSearch import find_ps_in_solution
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from LCS.PSEvaluator import GeneralPSEvaluator
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from utils import execution_timer


class ExplanationMiner:
    optimisation_problem: BenchmarkProblem
    ps_search_budget: int
    ps_search_population_size: int

    fs_evaluator: FSEvaluator
    ps_evaluator: GeneralPSEvaluator

    pRef: PRef

    verbose: bool

    preferred_culling_method: str

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 ps_search_budget: int,
                 ps_search_population: int,
                 pRef: PRef,
                 preferred_culling_method: str = "biggest",
                 verbose: bool = False):
        self.verbose = verbose
        self.preferred_culling_method = preferred_culling_method

        self.optimisation_problem = optimisation_problem
        self.ps_search_budget = ps_search_budget
        self.ps_search_population_size = ps_search_population
        self.pRef = pRef

        self.ps_evaluator = GeneralPSEvaluator(optimisation_problem=self.optimisation_problem, pRef=self.pRef)
        self.fs_evaluator = FSEvaluator(fitness_function=optimisation_problem.fitness_function)

    def find_pss(self,
                 main_solution: FullSolution,
                 unexplained_mask: np.ndarray,
                 culling_method: str,
                 proportion_unexplained_that_needs_used: Optional[float] = None,
                 proportion_used_that_should_be_unexplained: Optional[float] = None,) -> list[PS]:
        return find_ps_in_solution(to_explain=main_solution,
                                   unexplained_mask=unexplained_mask,
                                   population_size=self.ps_search_population_size,
                                   ps_evaluator=self.ps_evaluator,
                                   ps_budget=self.ps_search_budget,
                                   culling_method=culling_method,
                                   proportion_unexplained_that_needs_used=proportion_unexplained_that_needs_used,
                                   proportion_used_that_should_be_unexplained=proportion_used_that_should_be_unexplained,
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
        optima = self.pRef.get_best_solution()
        closest_to_optima = PRefManager.get_most_similar_solution_to(pRef=self.pRef,
                                                                     solution=optima)  # excludes the solution itself

        return self.consistency_test_on_solution_pair(optima, closest_to_optima,
                                                      culling_method=culling_method,
                                                      runs=runs)

    def get_accuracy_of_explanations_on_pair(self,
                                             main_solution: EvaluatedFS,
                                             background_solution: EvaluatedFS,
                                             p_value_tester: WilcoxonTest):

        with execution_timer() as timer:
            ps = self.find_pss(main_solution,
                               background_solution,
                               culling_method=self.preferred_culling_method)[0]

        beneficial_p_value, maleficial_p_value = p_value_tester.get_p_values_of_ps(ps)

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
            main_solution = self.pRef.get_nth_solution(index=random.randrange(self.pRef.sample_size))
            background_solution = PRefManager.get_most_similar_solution_to(pRef=self.pRef, solution=main_solution)
            return main_solution, background_solution

        mwu_tester = WilcoxonTest(sample_size=1000,
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
                                   descriptors_manager: DescriptorsManager) -> PS:

        pss = []
        for background_solution in tqdm(background_solutions):
            new_pss = self.find_pss(main_solution,
                                    background_solution,
                                    culling_method=self.preferred_culling_method)
            pss.extend(new_pss)

        print(
            f"For the solution \n\t{self.optimisation_problem.repr_fs(main_solution)}\n, and the {len(background_solutions)} background solutions:")
        for background_solution, pattern in zip(background_solutions, pss):
            description = descriptors_manager.get_descriptors_string(ps=pattern)
            print(f"background = \n{utils.indent(self.optimisation_problem.repr_fs(background_solution))}")
            print(f"pattern = \n{utils.indent(self.optimisation_problem.repr_ps(pattern))}")
            print(f"description = \n{utils.indent(description)}")
            print(f"\n")

        return pss[0]

    def get_background_solutions(self, main_solution: EvaluatedFS, background_solution_count: int) -> list[EvaluatedFS]:
        return PRefManager.get_most_similar_solutions_to(pRef=self.pRef,
                                                         solution=main_solution,
                                                         amount_to_return=background_solution_count)

    def get_temporary_descriptors_manager(self, control_samples_per_size_category: int = 1000) -> DescriptorsManager:
        pRef_manager = PRefManager(problem=self.optimisation_problem,
                                   pRef_file=None,
                                   instantiate_own_evaluator=False,
                                   verbose=True)
        pRef_manager.set_pRef(self.pRef)

        descriptors_manager = DescriptorsManager(optimisation_problem=self.optimisation_problem,
                                                 control_pss_file=None,
                                                 control_descriptors_table_file=None,
                                                 control_samples_per_size_category=control_samples_per_size_category,
                                                 speciality_threshold=0.5,
                                                 verbose=True)

        descriptors_manager.start_from_scratch()
        return descriptors_manager

    def get_random_explanation(self):
        solution_to_explain = self.pRef.get_nth_solution(index=random.randrange(self.pRef.sample_size))
        background_solutions = self.get_background_solutions(main_solution=solution_to_explain,
                                                             background_solution_count=5)

        descriptors_manager = self.get_temporary_descriptors_manager()
        self.produce_explanation_sample(main_solution=solution_to_explain,
                                        background_solutions=background_solutions,
                                        descriptors_manager=descriptors_manager)

    def get_weekday_score_of_solution(self, fs: FullSolution, weekday: str) -> float:
        assert (isinstance(self.optimisation_problem, EfficientBTProblem))
        fitness_breakdown = self.optimisation_problem.breakdown_of_fitness_function(fs)
        return fitness_breakdown["by_weekday"][weekday]

    def get_saturday_score_of_solution(self, fs: FullSolution) -> float:
        return self.get_weekday_score_of_solution(fs, "Saturday")

    def get_solutions_with_better_weekday(self, main_solution: FullSolution, weekday: str) -> list[FullSolution]:
        # they are also sorted by similarity
        assert (isinstance(self.optimisation_problem, EfficientBTProblem))

        solutions_and_satfits = [(solution, self.get_weekday_score_of_solution(solution, weekday))
                                 for solution in self.pRef.get_evaluated_FSs()]
        own_satfit = self.get_weekday_score_of_solution(main_solution, weekday)

        eligible_solutions = [solution
                              for solution, satfit in solutions_and_satfits
                              if own_satfit - satfit > 1e-05]  # also removes main_solution

        eligible_solutions.sort(key=lambda x: x.get_hamming_distance(main_solution))

        return eligible_solutions

    def get_partially_better_solutions(self, main_solution: FullSolution) -> list[
        (FullSolution, np.ndarray, np.ndarray)]:
        assert (isinstance(self.optimisation_problem, EfficientBTProblem))

        def partial_fitness_for_solution(solution: FullSolution) -> np.ndarray:
            partial_fit = self.optimisation_problem.breakdown_of_fitness_function(solution)
            return np.array([partial_fit["by_weekday"][weekday] for weekday in utils.weekdays])

        own_partial_fitness = partial_fitness_for_solution(main_solution)
        full_solutions = self.pRef.get_evaluated_FSs()
        pRef_partial_fitnesses = np.array([partial_fitness_for_solution(solution)
                                           for solution in full_solutions])

        better_value_matrix = (own_partial_fitness - pRef_partial_fitnesses) > 1e-05
        useful_rows = np.any(better_value_matrix, axis=1)

        wanted_solutions = [solution
                            for solution, is_good in zip(full_solutions, useful_rows)
                            if is_good]

        wanted_partial_fitnesses = pRef_partial_fitnesses[useful_rows]
        return list(zip(wanted_solutions, wanted_partial_fitnesses, better_value_matrix[useful_rows]))

    def get_explanation_to_improve_weekday(self,
                                           main_solution: FullSolution,
                                           weekday: str,
                                           descriptors_manager: DescriptorsManager) -> PairwiseExplanation:
        # partial_improvements = self.get_partially_better_solutions(main_solution)
        eligible_weekday_improvements = self.get_solutions_with_better_weekday(main_solution, weekday)
        if len(eligible_weekday_improvements) == 0:
            print(f"Seems that the solution already has the best {weekday}s...")
        background_solution = eligible_weekday_improvements[0]
        """ It would be interesting to plot the background solutions with
                x_axis = satfit
                y_axis = hamming_distance(main_solution)"""

        def prune_tradeoff(hamming_sat_data: list[(int, float)]) -> list[(int, float)]:
            all_hamming_distances = set(hamming for hamming, satfit in hamming_sat_data)
            best_for_hamming_distance_dict = dict()
            for hamming_distance_category in all_hamming_distances:
                best_for_hamming_distance_dict[hamming_distance_category] = min(
                    satfit for hamming, satfit in hamming_sat_data if hamming == hamming_distance_category)

            return list(best_for_hamming_distance_dict.items())

        interesting_data = [
            (alternative.get_hamming_distance(main_solution), self.get_weekday_score_of_solution(alternative, weekday))
            for alternative in eligible_weekday_improvements]

        pruned_interesting_data = prune_tradeoff(interesting_data)

        xs, ys = zip(*pruned_interesting_data)
        utils.simple_scatterplot(x_label="Hamming distance", y_label="Partial fitness", xs=xs, ys=ys)

        main_solution_satfit = self.get_weekday_score_of_solution(main_solution, weekday)
        background_satfit = self.get_weekday_score_of_solution(background_solution, weekday)

        print(
            f"You are intending to improve the {weekday} score of the known optima, which has satfit = {main_solution_satfit}")
        print(f"The background solution that has better satfit and is the closest "
              f"has hamming distance = {background_solution.get_hamming_distance(main_solution)},"
              f"and satfit = {background_satfit}")

        return self.get_pairwise_explanation(main_solution,
                                             background_solution,
                                             descriptors_manager)

    def get_pairwise_explanation(self,
                                 main_solution: FullSolution,
                                 background_solution: FullSolution,
                                 descriptor: DescriptorsManager) -> PairwiseExplanation:
        pss = self.find_pss(main_solution,
                            background_solution,
                            culling_method=self.preferred_culling_method)

        if len(pss) < 1:
            raise Exception("The pairwise explanation tester was not able to find a satisfactory explanation...")

        ps = pss[0]

        names_values_percentiles = descriptor.get_significant_descriptors_of_ps(ps)
        descriptor_string = descriptor.descriptors_tuples_into_string(names_values_percentiles, ps)

        in_main = PairwiseExplanation(main_solution,
                                      background_solution,
                                      ps,
                                      descriptor_tuples=names_values_percentiles,
                                      explanation_text=descriptor_string)

        return in_main

    def evaluate_explanation(self,
                             expl: PairwiseExplanation,
                             hypothesis_tester: WilcoxonTest,
                             near_optima_hypothesis_tester: WilcoxonNearOptima) -> dict:

        main_fitness, background_fitness = [self.optimisation_problem.fitness_function(s)
                                            for s in [expl.main_solution, expl.background_solution]]

        better_solution = ""
        if main_fitness > background_fitness:
            better_solution = "main"
        elif background_fitness > main_fitness:
            better_solution = "background"
        else:
            better_solution = "equivalent"

        greater_p_value, lower_p_value = hypothesis_tester.get_p_values_of_ps(expl.partial_solution)
        near_optima_greater_p_value, near_optima_lower_p_value = near_optima_hypothesis_tester.get_p_values_of_ps(
            expl.partial_solution)

        threshold = 0.05
        normal_effect = effect_string(greater_p_value, lower_p_value, threshold)
        near_optima_effect = effect_string(near_optima_greater_p_value, near_optima_lower_p_value, threshold)

        result = {"greater_p_value": greater_p_value,
                  "lower_p_value": lower_p_value,
                  "near_optima_greater_p_value": near_optima_greater_p_value,
                  "near_optima_lower_p_value": near_optima_lower_p_value,
                  "main_fitness": main_fitness,
                  "background_fitness": background_fitness,
                  "better_solution": better_solution,
                  "threshold": threshold,
                  "normal_effect": normal_effect,
                  "near_optima_effect": near_optima_effect,
                  "is_accurate": None, }

        match (better_solution, normal_effect, near_optima_effect):
            case ("main", _, "Positive") | ("background", _, "Negative"):
                result["is_accurate"] = True
            case (_, _, "Inconclusive"):
                result["is_accurate"] = False

        return result
