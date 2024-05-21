import os
import re
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import t, stats

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Explanation.MinedPSManager import MinedPSManager
from Explanation.PSPropertyManager import PSPropertyManager
from PSMiners.Mining import get_history_pRef
from utils import announce


class Detector:
    problem: BenchmarkProblem

    pRef_file: str   # npz
    mined_ps_manager: MinedPSManager  # manages some npz files
    ps_property_manager: PSPropertyManager   # which will manage a csv

    minimum_acceptable_ps_size: int
    verbose: bool


    cached_pRef: Optional[PRef]
    cached_pRef_mean: Optional[float]
    cached_pss: Optional[list[PS]]
    cached_control_pss: Optional[list[PS]]
    cached_properties: Optional[pd.DataFrame]

    search_metrics_evaluator: Optional[Classic3PSEvaluator]

    speciality_threshold: float

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 ps_file: str,
                 control_ps_file: str,
                 properties_file: str,
                 speciality_threshold: float,
                 minimum_acceptable_ps_size: int = 2,
                 verbose = False):
        self.problem = problem
        self.pRef_file = pRef_file
        self.mined_ps_manager = MinedPSManager(problem = problem,
                                               mined_ps_file=ps_file,
                                               control_ps_file=control_ps_file,
                                               verbose=verbose)
        self.ps_property_manager = PSPropertyManager(problem = problem,
                                                     property_table_file=properties_file,
                                                     verbose=verbose,
                                                     threshold=speciality_threshold)
        self.speciality_threshold = speciality_threshold
        self.minimum_acceptable_ps_size = minimum_acceptable_ps_size
        self.verbose = verbose

        self.cached_pRef = None
        self.cached_pRef_mean = None
        self.cached_pss = None
        self.cached_control_pss = None
        self.cached_properties = None
        self.search_metrics_evaluator = None


    @classmethod
    def from_folder(cls,
                    problem: BenchmarkProblem,
                    folder: str,
                    speciality_threshold = 0.1,
                    verbose = False):
        pRef_file = os.path.join(folder, "pRef.npz")
        ps_file = os.path.join(folder, "mined_ps.npz")
        control_ps_file = os.path.join(folder, "control_ps.npz")
        properties_file = os.path.join(folder, "ps_properties.csv")

        return cls(problem = problem,
                   pRef_file = pRef_file,
                   ps_file = ps_file,
                   control_ps_file = control_ps_file,
                   properties_file = properties_file,
                   speciality_threshold = speciality_threshold,
                   verbose=verbose)


    @property
    def pss(self):
        return self.mined_ps_manager.pss


    def set_cached_pRef(self, new_pRef: PRef):
        self.cached_pRef = new_pRef
        self.cached_pRef_mean = np.average(self.cached_pRef.fitness_array)
        self.search_metrics_evaluator = Classic3PSEvaluator(self.cached_pRef)

    def generate_pRef(self,
                      sample_size: int,
                      which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"]):

        with announce(f"Generating the PRef using {which_algorithm} and writing it to {self.pRef_file}", self.verbose):
            pRef  = get_history_pRef(benchmark_problem=self.problem,
                                     which_algorithm=which_algorithm,
                                     sample_size=sample_size,
                                     verbose=self.verbose)
        pRef.save(file=self.pRef_file)

        self.set_cached_pRef(pRef)

    @property
    def pRef(self) -> PRef:
        if self.cached_pRef is None:
            with announce(f"Loading the cached pRef from {self.pRef_file}"):
                self.set_cached_pRef(PRef.load(self.pRef_file))
        return self.cached_pRef

    @property
    def pRef_mean(self) -> float:
        if self.cached_pRef_mean is None:
            self.cached_pRef_mean = np.average(self.pRef.fitness_array)
        return self.cached_pRef_mean


    def t_test_for_mean_with_ps(self, ps: PS) -> (float, float):
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return -1, -1

        t_score = (sample_mean - self.cached_pRef_mean) / (sample_stdev / np.sqrt(n))
        p_value = 1 - t.cdf(abs(t_score), df=n - 1)
        return p_value, sample_mean


    def get_atomicity_contributions(self, ps: PS) -> np.ndarray:
        return self.search_metrics_evaluator.get_atomicity_contributions(ps, normalised=True)

    def get_average_when_present_and_absent(self, ps: PS) -> (float, float):
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        observations, not_observations = self.pRef.fitnesses_of_observations_and_complement(ps)
        return np.average(observations), np.average(not_observations)


    def get_fitness_delta_string(self, ps: PS) -> str:
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.get_average_when_present_and_absent(ps)
        delta = avg_when_present - avg_when_absent

        return (f"delta = {delta:.2f}, "
                 f"avg when present = {avg_when_present:.2f}, "
                 f"avg when absent = {avg_when_absent:.2f}")
        #f"p-value = {p_value:e}")


    def get_properties_string(self, ps: PS) -> str:
        pvrs = self.ps_property_manager.get_significant_properties_of_ps(ps)
        pvrs = self.ps_property_manager.sort_pvrs_by_rank(pvrs)
        return "\n".join(self.problem.repr_property(name, value, rank, ps)
                                   for name, value, rank in pvrs)

    def get_contributions_string(self, ps: PS) -> str:
        contributions = -self.get_atomicity_contributions(ps)
        return "contributions: " + utils.repr_with_precision(contributions, 2)



    def get_ps_description(self, ps: PS) -> str:
        return utils.indent("\n".join([self.problem.repr_extra_ps_info(ps),
                                       self.get_fitness_delta_string(ps),
                                       self.get_properties_string(ps)]))



    def get_best_n_full_solutions(self, n: int) -> list[EvaluatedFS]:
        solutions = self.pRef.get_evaluated_FSs()
        solutions.sort(reverse=True)
        return solutions[:n]

    @staticmethod
    def only_non_obscured_pss(pss: list[PS]) -> list[PS]:
        def obscures(ps_a: PS, ps_b: PS):
            a_fixed_pos = set(ps_a.get_fixed_variable_positions())
            b_fixed_pos = set(ps_b.get_fixed_variable_positions())
            if a_fixed_pos == b_fixed_pos:
                return False
            return b_fixed_pos.issubset(a_fixed_pos)

        def get_those_that_are_not_obscured_by(ps_list: PS, candidates: set[PS]) -> set[PS]:
            return {candidate for candidate in candidates if not obscures(ps_list, candidate)}

        current_candidates = set(pss)

        for ps in pss:
            current_candidates = get_those_that_are_not_obscured_by(ps, current_candidates)

        return list(current_candidates)


    def get_contained_ps(self, solution: EvaluatedFS, must_contain: Optional[int] = None) -> list[EvaluatedPS]:
        contained = [ps
                    for ps in self.pss
                    if contains(solution.full_solution, ps)
                    if ps.fixed_count() >= self.minimum_acceptable_ps_size]

        if must_contain is not None:
            contained = [ps for ps in contained
                         if ps[must_contain] != STAR]

        return contained

    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: int, must_contain: Optional[int] = None):
        contained_pss: list[EvaluatedPS] = self.get_contained_ps(solution, must_contain = must_contain)

        contained_pss = self.mined_ps_manager.sort_by_atomicity(contained_pss)

        print(f"The solution \n {utils.indent(self.problem.repr_fs(solution.full_solution))}\ncontains the following PSs:")
        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps)))
            print()



    def explanation_loop(self,
                         amount_of_fs_to_propose: int = 6,
                         ps_show_limit: int = 12,
                         show_debug_info = False,
                         show_global_properties = False):
        solutions = self.get_best_n_full_solutions(amount_of_fs_to_propose)

        print(f"The top {amount_of_fs_to_propose} solutions are")
        for solution in solutions:
            print(self.problem.repr_fs(solution.full_solution))
            print()

        if show_debug_info:
            self.describe_global_information(show_global_properties = show_global_properties)


        first_round = True

        while True:
            if first_round:
                print("Would you like to see some explanations of the solutions? Write an index, or n to exit")
            else:
                print("Type another index, or n to exit")
            answer = input().upper()
            if answer == "N":
                break
            elif answer.startswith("V:"):
                if "S" in list(answer):
                    try:
                        variable_index, solution_index = [int(s) for s in re.findall(r'\d+', answer)]

                        solution_to_explain = solutions[solution_index]
                        self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit, must_contain = variable_index)
                    except ValueError:
                        print("That didn't work, please retry")
                        continue
                else:
                    try:
                        variable_index = int(answer[2:])
                    except ValueError:
                        print("That didn't work, please retry")
                        continue
                    self.describe_properties_of_variable(variable_index)
            else:
                try:
                    index = int(answer)
                except ValueError:
                    print("That didn't work, please retry")
                    continue
                solution_to_explain = solutions[index]
                self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit)



    def generate_files_with_default_settings(self, pRef_size: Optional[int] = 10000, pss_budget: Optional[int] = 10000):

        self.generate_pRef(sample_size=pRef_size,
                           which_algorithm="SA")

        self.mined_ps_manager.generate_ps_file(pRef = self.pRef,
                                               ps_miner_method="sequential",
                                               ps_budget=pss_budget)
        self.mined_ps_manager.generate_control_pss_file(samples_for_each_category=1000)

        self.ps_property_manager.generate_property_table_file(self.mined_ps_manager.pss, self.mined_ps_manager.control_pss)


    def get_coverage_stats(self) -> np.ndarray:
        def ps_to_fixed_values_tally(ps: PS) -> np.ndarray:
            return ps.values != STAR

        return sum(ps_to_fixed_values_tally(ps) for ps in self.pss) / len(self.pss)


    def get_ps_size_distribution(self):
        sizes = [ps.fixed_count() for ps in self.pss]
        unique_sizes = sorted(list(set(sizes)))
        def proportion_for_size(target_size: int) -> float:
            return len([1 for item in sizes if item == target_size]) / len(sizes)

        return {size: proportion_for_size(size)
                for size in unique_sizes}


    def describe_global_information(self, show_global_properties = False):
        print("The partial solutions cover the search space with the following distribution:")
        print(utils.repr_with_precision(self.get_coverage_stats(), 2))

        print("The distribution of PS sizes is")
        distribution = self.get_ps_size_distribution()
        print("\t"+"\n\t".join(f"{size}: {int(prop*100)}%" for size, prop in distribution.items()))

        if show_global_properties:
            self.print_global_properties()



    def get_variable_properties(self, var_index: int, value: Optional[int] = None) -> dict:

        # TODO think about this more thoroughly
        # should we compare against control PSs or experimental PSs?



        if value is None:
            which_pss_contain_var = [var_index in ps.get_fixed_variable_positions()
                                     for ps in self.pss]
        else:
            which_pss_contain_var = [ps[var_index] == value
                                     for ps in self.pss]
        relevant_properties = self.properties[self.properties["control"]==False][which_pss_contain_var]
        relevant_properties = relevant_properties[relevant_properties["size"] > 1]
        control_properties = self.properties[self.properties["control"]==True]


        def valid_column_values_from(df: pd.DataFrame, column_name):
            """ This is why I hate pandas"""
            column = df[column_name].copy()
            column.dropna(inplace=True)
            column = column[~np.isnan(column)]
            """ this tiny snipped took me half an hour, by the way. Modify with care"""
            return column.values

        def p_value_of_difference_of_means(property_name: str) -> float:
            experimental_values = valid_column_values_from(relevant_properties, property_name)
            control_values = valid_column_values_from(control_properties, property_name)

            if len(experimental_values) < 2 or len(control_values) < 2:
                return 1.0
            t_value, p_value = stats.ttest_ind(experimental_values, control_values)
            return p_value

        properties_and_p_values = {prop: p_value_of_difference_of_means(prop)
                                   for prop in control_properties.columns
                                   if prop != "size"
                                   if prop != "control"}

        return properties_and_p_values


    def get_variables_properties_table(self):
        output_file_name = self.ps_file[:-4]+"_variables.csv"
        dicts = [self.get_variable_properties(i) for i in range(self.problem.search_space.amount_of_parameters)]
        df = pd.DataFrame(dicts)
        with announce(f"Writing the variable data to {output_file_name}"):
            df.to_csv(output_file_name, index=False)


    def describe_properties_of_variable(self, var: int, value: Optional[int] = None):

        if value is None:
            properties = [(prop, p_value) for prop, p_value in self.get_variable_properties(var).items()
                          if p_value < 0.05]
        else:
            properties = [(prop, p_value) for prop, p_value in self.get_variable_properties(var, value).items()
                          if p_value < 0.05]
        properties.sort(key=utils.second)

        if value is None:
            print(f"Significant properties for the variable {var}:")
        else:
            print(f"Significant properties for the variable {var} when it's = {value}:")
        for prop, p_value in properties:
            print(f"\t{prop}, with p-value {p_value:e}")





















