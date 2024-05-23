import os
import re
from typing import Optional

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Explanation.MinedPSManager import MinedPSManager
from Explanation.PRefManager import PRefManager
from Explanation.PSPropertyManager import PSPropertyManager


class Detector:
    problem: BenchmarkProblem

    pRef_manager: PRefManager   # manages a npz
    mined_ps_manager: MinedPSManager  # manages some npz files
    ps_property_manager: PSPropertyManager   # which will manage a csv

    minimum_acceptable_ps_size: int
    verbose: bool

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
        self.pRef_manager = PRefManager(problem = problem,
                                        pRef_file = pRef_file)
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
    def pss(self) -> list[EvaluatedPS]:
        return self.mined_ps_manager.pss


    @property
    def pRef(self) -> PRef:
        return self.pRef_manager.pRef



    def get_fitness_delta_string(self, ps: PS) -> str:
        p_value, _ = self.pRef_manager.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.pRef_manager.get_average_when_present_and_absent(ps)
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
        contributions = -self.pRef_manager.get_atomicity_contributions(ps)
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


    def sort_pss(self, pss: list[EvaluatedPS]) -> list[EvaluatedPS]:
        def get_atomicity(ps: EvaluatedPS) -> float:
            return ps.metric_scores[2]

        def get_mean_fitness(ps: EvaluatedPS) -> float:
            return ps.metric_scores[1]

        def get_simplicity(ps: EvaluatedPS) -> float:
            return ps.metric_scores[0]

        return utils.sort_by_combination_of(pss, key_functions=[get_mean_fitness, get_atomicity], reverse=False)

    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: int, must_contain: Optional[int] = None):
        contained_pss: list[EvaluatedPS] = self.get_contained_ps(solution, must_contain = must_contain)


        def get_delta(ps: PS) -> float:
            avg_when_present, avg_when_absent = self.pRef_manager.get_average_when_present_and_absent(ps)
            return avg_when_present - avg_when_absent

        #contained_pss.sort(key=get_delta, reverse=True)   # sort by delta
        #contained_pss = self.mined_ps_manager.sort_by_atomicity(contained_pss)  # sort by atomicity
        contained_pss = self.sort_pss(contained_pss)

        print(f"The solution \n"
              f"{utils.indent(self.problem.repr_fs(solution.full_solution))}\n"
              f"contains the following PSs:")
        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps)))
            print()



    def explanation_loop(self,
                         amount_of_fs_to_propose: int = 6,
                         ps_show_limit: int = 12,
                         show_debug_info = False):
        solutions = self.get_best_n_full_solutions(amount_of_fs_to_propose)

        print(f"The top {amount_of_fs_to_propose} solutions are")
        for solution in solutions:
            print(self.problem.repr_fs(solution.full_solution))
            print()

        if show_debug_info:
            self.describe_global_information()


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
            elif answer.startswith("PLOTVAR"):
                try:
                    variable_index = int(input("Which variable?"))
                    property_name = input("Which property?")


                    self.ps_property_manager.plot_var_property(var_index=variable_index,
                                                               value=None,
                                                               property_name=property_name,
                                                               pss=self.pss)
                except ValueError:
                    print("That didn't work, please retry")
                    continue
            else:
                try:
                    index = int(answer)
                except ValueError:
                    print("That didn't work, please retry")
                    continue
                solution_to_explain = solutions[index]
                self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit)



    def generate_files_with_default_settings(self, pRef_size: Optional[int] = 10000, pss_budget: Optional[int] = 10000):

        self.pRef_manager.generate_pRef_file(sample_size=pRef_size,
                                             which_algorithm="SA")

        self.mined_ps_manager.generate_ps_file(pRef = self.pRef,
                                               ps_miner_method="sequential",
                                               ps_budget=pss_budget)
        self.mined_ps_manager.generate_control_pss_file(samples_for_each_category=1000)

        self.ps_property_manager.generate_property_table_file(self.mined_ps_manager.pss, self.mined_ps_manager.control_pss)





    def get_ps_size_distribution(self):
        sizes = [ps.fixed_count() for ps in self.pss]
        unique_sizes = sorted(list(set(sizes)))
        def proportion_for_size(target_size: int) -> float:
            return len([1 for item in sizes if item == target_size]) / len(sizes)

        return {size: proportion_for_size(size)
                for size in unique_sizes}


    def describe_global_information(self):
        print("The partial solutions cover the search space with the following distribution:")
        print(utils.repr_with_precision(self.mined_ps_manager.get_coverage_stats(), 2))

        print("The distribution of PS sizes is")
        distribution = self.get_ps_size_distribution()
        print("\t"+"\n\t".join(f"{size}: {int(prop*100)}%" for size, prop in distribution.items()))


    def describe_properties_of_variable(self, var: int, value: Optional[int] = None):

        properties = [(prop, p_value, prop_mean, control_mean)
                      for prop, (p_value, prop_mean, control_mean) in self.ps_property_manager.get_variable_properties_stats(self.pss, var, value).items()
                      if p_value < 0.05]

        properties.sort(key=utils.second)

        if value is None:
            print(f"Significant properties for the variable {var}:")
        else:
            print(f"Significant properties for the variable {var} when it's = {value}:")
        for prop, p_value, prop_mean, control_mean in properties:
            print(f"\t{prop}, with p-value {p_value:e}, prop_mean = {prop_mean:.2f}, control_mean = {control_mean:.2f}")





















