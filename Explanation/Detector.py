import itertools
import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import t

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
            termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget)
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
        control_pss = generate_control_PSs(self.problem.search_space, reference_pss=self.pss)
        write_pss_to_file(control_pss, self.control_ps_file)
        self.cached_control_pss = control_pss


    @property
    def control_pss(self) -> list[PS]:
        if self.cached_control_pss is None:
            with announce(f"Loading the control pss from {self.ps_file}", self.verbose):
                self.cached_control_pss = load_pss(self.control_ps_file)
        return self.cached_control_pss

    def generate_properties_csv_file(self):

        with announce(f"Generating the properties file and storing it at {self.properties_file}", self.verbose):
            properties_dicts = [self.problem.ps_to_properties(ps) for ps in itertools.chain(self.pss, self.control_pss)]
            properties_df = pd.DataFrame(properties_dicts)
            properties_df["control"] = np.array([index > len(self.pss) for index in range(len(properties_dicts))])   # not my best work

            properties_df.to_csv(self.properties_file, index=False)
        self.cached_properties = properties_df

    @property
    def properties(self):
        if self.cached_properties is None:
            with announce(f"Loading the properties from {self.properties_file}", self.verbose):
                self.cached_properties = pd.read_csv(self.properties_file)
        return self.cached_properties


    def relative_property_ranking_within_dataset(self, property_name: str, property_value) -> (float, float):
        known_values = list(self.properties[property_name])
        known_values = [value for value in known_values if not np.isnan(value)]
        known_values.sort()


        index_just_before = 0
        while index_just_before < len(known_values) and known_values[index_just_before] < property_value :
            index_just_before +=1

        index_just_after = index_just_before
        while index_just_after < len(known_values) and known_values[index_just_after] == property_value:
            index_just_after += 1

        total_quantity = len(known_values)
        return index_just_before / total_quantity, index_just_after / total_quantity



    def relative_property_rank_is_significant(self, position: (float, float)) -> Optional[str]:
        lower_bound, upper_bound = position
        is_low = upper_bound < self.speciality_threshold
        is_high = lower_bound > (1-self.speciality_threshold)

        if is_low:
            return "less"
        elif is_high:
            return "more"
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


    def only_significant_properties(self, all_properties: dict) -> list[(str, float, float)]:
        # obtain the relative_property_rankings
        items = [(key, value, self.relative_property_ranking_within_dataset(key, value))
                 for key, value in all_properties.items()
                 if key != "control"]

        # only keep the ones that are significant
        items = [(key, value, relative_property_rank) for (key, value, relative_property_rank) in items
                 if self.relative_property_rank_is_significant(relative_property_rank)]
        return items

    def get_ps_description(self, ps: PS, ps_properties: dict) -> str:
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.get_average_when_present_and_absent(ps)
        delta = avg_when_present - avg_when_absent
        significant_properties = self.only_significant_properties(ps_properties)

        def get_rank_significance(kvr):
            lower, upper = kvr[2]
            pillar = lower if lower < 1-upper else upper
            return abs(pillar-0.5)*2

        significant_properties.sort(key=get_rank_significance, reverse=True)



        avg_with_and_without_str =  (f"delta = {delta:.2f}, "
                                     f"avg when present = {avg_when_present:.2f}, "
                                     f"avg when absent = {avg_when_absent:.2f}")
                                     #f"p-value = {p_value:e}")

        contributions = -self.get_atomicity_contributions(ps)
        contribution_str = "contributions: " + utils.repr_with_precision(contributions, 2)

        def repr_property(kvr) -> str:
            key, value, rank_range = kvr
            return self.problem.repr_property(key, value, rank_range)


        properties_str = "\n".join(repr_property(kvr) for kvr in significant_properties)

        ps_details = self.problem.repr_extra_ps_info(ps)

        return utils.indent("\n".join([ps_details, avg_with_and_without_str, properties_str, contribution_str]))



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


    def get_contained_ps_with_properties(self, solution: EvaluatedFS) -> list[PSWithProperties]:
        def pd_row_to_dict(row):
            return dict(row[1])

        return [PSWithProperties(ps, pd_row_to_dict(properties))
                         for (ps, properties) in zip(self.pss, self.properties.iterrows())
                         if contains(solution.full_solution, ps)
                         if ps.fixed_count() >= self.minimum_acceptable_ps_size]






    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: int):
        contained_pss = self.get_contained_ps_with_properties(solution)

        #contained_pss = Explainer.only_non_obscured_pss(contained_pss)
        contained_pss.sort(reverse=True, key = lambda x: x.metric_scores[-1])  # sort by atomicity

        fs_as_ps = PS.from_FS(solution.full_solution)
        print(f"The solution \n {utils.indent(self.problem.repr_ps(fs_as_ps))}\ncontains the following PSs:")
        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps, ps.properties)))
            print()



    def explanation_loop(self,
                         amount_of_fs_to_propose: int = 6,
                         ps_show_limit: int = 12):
        solutions = self.get_best_n_full_solutions(amount_of_fs_to_propose)

        print(f"The top {amount_of_fs_to_propose} solutions are")
        for solution in solutions:
            print(self.problem.repr_fs(solution.full_solution))
            print()


        self.describe_global_information()


        first_round = True

        while True:
            if first_round:
                print("Would you like to see some explanations of the solutions? Write an index, or n to exit")
            else:
                print("Type another index, or n to exit")
            answer = input()
            if answer.upper() == "N":
                break
            else:
                try:
                    index = int(answer)
                except ValueError:
                    print("That didn't work, please retry")
                    continue
                solution_to_explain = solutions[index]
                self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit)



    def generate_files_with_default_settings(self):

        self.generate_pRef(sample_size=20000,
                           which_algorithm="SA")

        self.generate_pss(ps_miner_method="sequential",
                          ps_budget = 30000)

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


    def describe_global_information(self):
        print("The partial solutions cover the search space with the following distribution:")
        print(utils.repr_with_precision(self.get_coverage_stats(), 2))

        print("The distribution of PS sizes is")
        distribution = self.get_ps_size_distribution()
        print("\t"+"\n\t".join(f"{size}: {int(prop*100)}%" for size, prop in distribution.items()))


    def get_ps_which_cover_any(self, solution: EvaluatedFS, vars: set[int]) -> list[(PS, set[int])]:
        def get_contained_special_vars(ps: PS):
            fixed_vars = ps.get_fixed_variable_positions()
            intersection = [var for var in vars if var in fixed_vars]
            return intersection

        contained_pss = self.get_contained_ps_with_properties(solution)
        contained_pss = [(ps, get_contained_special_vars(ps)) for ps in contained_pss]
        contained_pss = [(ps, intersection)
                         for ps, intersection in contained_pss
                         if len(intersection) > 0]
        return contained_pss
    def explain_contained_ps_which_are_affected(self, solution: EvaluatedFS, vars: set):
        ps_with_vars = self.get_ps_which_cover_any(solution, vars)
        ps_with_vars.sort(reverse=True, key = lambda x: len([1]))  # the ones with the biggest intersection first


        print(f"The affected partial solutions are")
        shown_ps_max = 10
        for ps, intersection in ps_with_vars[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps, ps.properties)))
            print(f"(Conflicts = {intersection})")
            print()

    def get_patches_usable_to_cover(self, original_solution: EvaluatedFS, vars_to_change: set[int]) -> list[EvaluatedPS]:
        def patch_can_be_used(patch: (PS, set[int]))-> bool:
            ps, vars_used = patch
            for variable in vars_used:
                value = ps[variable]
                if value == original_solution.values[variable]:
                    return False
            return True


        candidate_patches = self.get_ps_which_cover_any(original_solution, vars_to_change)
        candidate_patches = filter(patch_can_be_used, candidate_patches)

        candidate_patches = [EvaluatedPS(ps.values) for ps in candidate_patches]
        for candidate in candidate_patches:
            simplicity, mean_fitness, atomicity = self.search_metrics_evaluator.get_S_MF_A(candidate)
            candidate.aggregated_score = mean_fitness

        return candidate_patches




    def patch_fs(self, original_solution: EvaluatedFS, vars_to_change: set[int]) -> EvaluatedFS:
        patches = self.get_patches_usable_to_cover(original_solution, vars_to_change)
        sampler = PickAndMergeSampler(self.problem.search_space,
                                      individuals=patches)
        attemps = 10
        incomplete_ps = PS.from_FS(original_solution)
        for var in vars_to_change:
            incomplete_ps.values[var] = STAR


        resulting_fss = []
        for attempt in range(attemps):
            new_fs = sampler.apply_patches(starting_point=incomplete_ps, original_FS=original_solution)
            resulting_fss.append(EvaluatedFS(new_fs, self.problem.fitness_function(new_fs)))

        if len(resulting_fss) == 0:
            raise Exception("Could not resolve using patches")
        resulting_fss.sort(reverse=True)
        return resulting_fss[0]

    def find_alternative_solution(self,
                                  original_solution: EvaluatedFS,
                                  vars_to_change: set[int],
                                  method: str) -> EvaluatedPS:
        if method == "patch_ps":
            return self.patch_fs(original_solution, vars_to_change)
        else:
            raise Exception("The non-exhaustive search has not been implemented yet")

    def request_change(self,
                       solution: EvaluatedFS,
                       vars_to_change: set,
                       method: str):
        """ In the event of the decision maker not being happy with certain variables of a full solution, we can suggest
            * the partial solutions that will be affected
            * best replacement values, and the would be fitness
        """

        print(f"The fitness of the solution is {solution.fitness:.3f}")
        self.explain_contained_ps_which_are_affected(solution, vars_to_change)





















