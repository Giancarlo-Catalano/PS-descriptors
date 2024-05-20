import itertools
import os
import re
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import t, stats

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedFS import EvaluatedFS
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.PickAndMerge import PickAndMergeSampler
from Explanation.detection_utils import generate_control_PSs
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.Mining import get_history_pRef, get_ps_miner, write_evaluated_pss_to_file, load_pss, write_pss_to_file
from utils import announce

class PSWithProperties(EvaluatedPS):
    properties: dict

    def __init__(self, ps: EvaluatedPS, properties: dict):
        super().__init__(ps.values, metric_scores=ps.metric_scores)
        self.properties = properties


class Detector:
    """usage: this class requires you to make some files first, and then you can use it by loading those files"""
    """ 
            Run this first
            BTDetector = BTDetector(problem = problem,
                          folder=experimental_directory,
                          speciality_threshold=0.1,
                          verbose=True)
            BTDetector.generate_files_with_default_settings()
            
            
            Then you can just do this to run the explainer as many times as you want:
            BTDetector = BTDetector(problem = problem,
                          folder=experimental_directory,
                          speciality_threshold=0.1,
                          verbose=True)

            BTDetector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)
            
            
            BTDetector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)
    """

    problem: BenchmarkProblem

    pRef_file: str   # npz
    ps_file: str     # npz
    control_ps_file: str   # npz
    properties_file: str   # csv

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
        self.ps_file = ps_file
        self.control_ps_file = control_ps_file
        self.properties_file = properties_file
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


    def generate_pss(self,
                     ps_miner_method : Literal["classic", "NSGA", "NSGA_experimental_crowding", "SPEA2", "sequential"] = "NSGA_experimental_crowding",
                     ps_budget: int = 10000):

        algorithm = get_ps_miner(self.pRef, which=ps_miner_method)

        with announce(f"Running {algorithm} on {self.pRef} with {ps_budget =}", self.verbose):
            budget_limit = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget)
            coverage_limit = TerminationCriteria.SearchSpaceIsCovered()
            termination_criterion = TerminationCriteria.UnionOfCriteria(budget_limit, coverage_limit)
            algorithm.run(termination_criterion, verbose=self.verbose)

        result_ps = algorithm.get_results(None)
        result_ps = AbstractPSMiner.without_duplicates(result_ps)
        result_ps = [ps for ps in result_ps if not ps.is_empty()]

        with announce(f"Writing the PSs onto {self.ps_file}"):
            write_evaluated_pss_to_file(result_ps, self.ps_file)

        self.cached_pss = result_ps

    @property
    def pss(self) -> list[PS]:
        if self.cached_pss is None:
            with announce(f"Loading the cached pss from {self.ps_file}"):
                self.cached_pss = load_pss(self.ps_file)
        return self.cached_pss

    def generate_control_pss(self):
        control_pss = generate_control_PSs(self.problem.search_space,
                                           reference_pss=self.pss,
                                           samples_for_each_category=1000)
        write_pss_to_file(control_pss, self.control_ps_file)
        self.cached_control_pss = control_pss


    @property
    def control_pss(self) -> list[PS]:
        if self.cached_control_pss is None:
            with announce(f"Loading the control pss from {self.control_ps_file}", self.verbose):
                self.cached_control_pss = load_pss(self.control_ps_file)
        return self.cached_control_pss

    def generate_properties_csv_file(self):
        # only use the control pss

        with announce(f"Generating the properties file and storing it at {self.properties_file}", self.verbose):
            properties_dicts = [self.problem.ps_to_properties(ps) for ps in itertools.chain(self.pss, self.control_pss)]
            properties_df = pd.DataFrame(properties_dicts)
            properties_df["control"] = np.array([index >= len(self.pss) for index in range(len(properties_dicts))])   # not my best work
            properties_df["size"] = np.array([ps.fixed_count() for ps in itertools.chain(self.pss, self.control_pss)])

            properties_df.to_csv(self.properties_file, index=False)
        self.cached_properties = properties_df

    @property
    def properties(self):
        if self.cached_properties is None:
            with announce(f"Loading the properties from {self.properties_file}", self.verbose):
                self.cached_properties = pd.read_csv(self.properties_file)
        return self.cached_properties


    def relative_property_ranking_within_dataset(self, ps: PS, property_name: str, property_value) -> float:

        order_of_ps = ps.fixed_count()
        is_control = self.properties["control"] == True
        is_same_size = self.properties["size"] == order_of_ps
        control_rows = self.properties[is_control & is_same_size]
        control_values = control_rows[property_name]
        control_values = [value for value in control_values if not np.isnan(value)]

        return utils.ecdf(property_value, control_values)



    def relative_property_rank_is_significant(self, position: float) -> Optional[str]:
        if position < self.speciality_threshold:
            return "low"
        elif (1-position) < self.speciality_threshold:
            return "high"
        else:
            return None


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


    def only_significant_properties(self, ps: PS, all_properties: dict) -> list[(str, float, float)]:
        # obtain the relative_property_rankings
        items = [(key, value, self.relative_property_ranking_within_dataset(ps, key, value))
                 for key, value in all_properties.items()
                 if key != "control"
                 if key != "size"]

        # only keep the ones that are significant
        items = [(key, value, relative_property_rank) for (key, value, relative_property_rank) in items
                 if self.relative_property_rank_is_significant(relative_property_rank)]
        return items


    @classmethod
    def get_rank_significance(cls, kvr):
        rank = kvr[2]
        return abs(rank-0.5)*2

    def get_ps_description(self, ps: PS, ps_properties: dict) -> str:
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.get_average_when_present_and_absent(ps)
        delta = avg_when_present - avg_when_absent

        # obtain the ranks
        properties_values_significances = [(name, value, self.relative_property_ranking_within_dataset(ps, name, value))
                                           for name, value in ps_properties.items()
                                           if name != "size"
                                           if name != "control"]

        # only keep the significant ones
        properties_values_significances = [(name, value, rank)
                                           for name, value, rank in properties_values_significances
                                           if self.relative_property_rank_is_significant(rank) is not None]

        # sort by importance
        properties_values_significances.sort(key=self.get_rank_significance, reverse=True)

        avg_with_and_without_str =  (f"delta = {delta:.2f}, "
                                     f"avg when present = {avg_when_present:.2f}, "
                                     f"avg when absent = {avg_when_absent:.2f}")
                                     #f"p-value = {p_value:e}")

        #contributions = -self.get_atomicity_contributions(ps)
        #contribution_str = "contributions: " + utils.repr_with_precision(contributions, 2)

        def repr_property(kvr) -> str:
            key, value, rank_range = kvr
            result = self.problem.repr_property(key, value, rank_range, ps)
            return result


        properties_str = "\n".join(repr_property(kvr) for kvr in properties_values_significances)

        ps_details = self.problem.repr_extra_ps_info(ps)

        return utils.indent("\n".join([ps_details, avg_with_and_without_str, properties_str]))



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


    def get_contained_ps_with_properties(self, solution: EvaluatedFS, must_contain: Optional[int] = None) -> list[PSWithProperties]:
        def pd_row_to_dict(row):
            return dict(row[1])

        if must_contain is not None:
            return [PSWithProperties(ps, pd_row_to_dict(properties))
                    for (ps, properties) in zip(self.pss, self.properties.iterrows())
                    if contains(solution.full_solution, ps)
                    if ps[must_contain] != STAR
                    if ps.fixed_count() >= self.minimum_acceptable_ps_size]
        else:
            return [PSWithProperties(ps, pd_row_to_dict(properties))
                             for (ps, properties) in zip(self.pss, self.properties.iterrows())
                             if contains(solution.full_solution, ps)
                             if ps.fixed_count() >= self.minimum_acceptable_ps_size]






    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: int, must_contain: Optional[int] = None):
        contained_pss = self.get_contained_ps_with_properties(solution, must_contain = must_contain)


        #contained_pss = Explainer.only_non_obscured_pss(contained_pss)
        contained_pss.sort(reverse=False, key = lambda x: len(x.properties))  # sort by atomicity

        fs_as_ps = PS.from_FS(solution.full_solution)
        print(f"The solution \n {utils.indent(self.problem.repr_ps(fs_as_ps))}\ncontains the following PSs:")
        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps, ps.properties)))
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

        self.generate_pss(ps_miner_method="sequential",
                          ps_budget = pss_budget)

        self.generate_control_pss()

        self.generate_properties_csv_file()


    def update_properties(self):
        self.generate_properties_csv_file()



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





















