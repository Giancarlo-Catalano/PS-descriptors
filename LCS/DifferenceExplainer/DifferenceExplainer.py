import os
from typing import Optional

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Explanation.PRefManager import PRefManager
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from LCS.LCSManager import LCSManager


class DifferenceExplainer:
    problem: BenchmarkProblem

    pRef_manager: PRefManager
    descriptors_manager: DescriptorsManager

    verbose: bool
    speciality_threshold: float

    lcs_manager: LCSManager

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 control_ps_file: str,
                 descriptors_file: str,
                 condition_ps_file: str,
                 rule_attribute_file: str,
                 speciality_threshold: float,
                 verbose=False):
        self.problem = problem
        self.pRef_manager = PRefManager(problem=problem,
                                        pRef_file=pRef_file,
                                        verbose=True)

        self.pRef_manager.load_from_existing_if_possible()

        self.descriptors_manager = DescriptorsManager(optimisation_problem=problem,
                                                      control_pss_file=control_ps_file,
                                                      control_descriptors_table_file=descriptors_file,
                                                      control_samples_per_size_category=1000,
                                                      verbose=verbose)

        self.descriptors_manager.load_from_existing_if_possible()

        self.lcs_manager = LCSManager(optimisation_problem=problem,
                                      rule_conditions_file=condition_ps_file,
                                      rule_attributes_file=rule_attribute_file,
                                      pRef = self.pRef)
        self.lcs_manager.load_from_existing_if_possible()

        self.speciality_threshold = speciality_threshold
        self.verbose = verbose

    @classmethod
    def from_folder(cls,
                    problem: BenchmarkProblem,
                    folder: str,
                    speciality_threshold=0.1,
                    verbose=False):
        pRef_file = os.path.join(folder, "pRef.npz")
        ps_file = os.path.join(folder, "mined_ps.npz")
        control_ps_file = os.path.join(folder, "control_ps.npz")
        descriptors_file = os.path.join(folder, "descriptors.csv")
        rule_attribute_file = os.path.join(folder, "rule_attributes.csv")

        return cls(
            problem=problem,
            pRef_file=pRef_file,
            control_ps_file=control_ps_file,
            descriptors_file=descriptors_file,
            condition_ps_file=ps_file,
            rule_attribute_file=rule_attribute_file,
            speciality_threshold=speciality_threshold,
            verbose=verbose)

    @property
    def pss(self) -> list[PS]:
        return self.lcs_manager.get_pss_from_model()

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
        # f"p-value = {p_value:e}")

    def get_significant_descriptors_of_ps(self, ps: PS) -> list[(str, float, float)]:
        descriptors = self.descriptors_manager.get_descriptors_of_ps(ps)
        size = ps.fixed_count()
        percentiles = self.descriptors_manager.get_percentiles_for_descriptors(ps_size=size, ps_descriptors=descriptors)

        names_values_percentiles = [(name, descriptors[name], percentiles[name]) for name in percentiles]

        # then we only consider values which are worth reporting
        def percentile_is_significant(percentile: float) -> bool:
            return (percentile < self.speciality_threshold) or (percentile > (1 - self.speciality_threshold))

        names_values_percentiles = [(name, value, percentile)
                                    for name, value, percentile in names_values_percentiles
                                    if percentile_is_significant(percentile)]

        # sort by "extremeness"
        names_values_percentiles.sort(key=lambda x: abs(0.5 - x.percentile), reverse=True)
        return names_values_percentiles

    def get_properties_string(self, ps: PS) -> str:
        names_values_percentiles = self.get_significant_descriptors_of_ps(ps)
        return "\n".join(self.problem.repr_property(name, value, percentile, ps)
                         for name, value, percentile in names_values_percentiles)

    def get_ps_description(self, ps: PS) -> str:
        return utils.indent("\n".join([self.problem.repr_extra_ps_info(ps),
                                       self.get_fitness_delta_string(ps),
                                       self.get_properties_string(ps)]))

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

    def get_contained_ps(self, solution: EvaluatedFS, must_contain: Optional[int] = None) -> list[PS]:
        contained = [ps
                     for ps in self.pss
                     if contains(solution.full_solution, ps)]

        if must_contain is not None:
            contained = [ps for ps in contained
                         if ps[must_contain] != STAR]

        return contained

    def get_difference_ps(self, solution_a: EvaluatedFS, solution_b: EvaluatedFS) -> (list[PS], list[PS]):
        in_a = []
        in_b = []
        for ps in self.pss:
            if contains(solution_a, ps):
                if not contains(solution_b, ps):
                    in_a.append(ps)
            elif contains(solution_b, ps):
                if not contains(solution_a, ps):
                    in_b.append(ps)

        return in_a, in_b

    def get_explainability_percentage_of_solution(self, pss: list[PS]) -> float:
        if len(pss) == 0:
            return 0
        used_vars = np.array([ps.values != STAR for ps in pss])
        used_vars_count = used_vars.sum(axis=0, dtype=bool).sum(dtype=int)
        return used_vars_count / len(pss[0])

    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: Optional[int] = None,
                         must_contain: Optional[int] = None):
        contained_pss: list[PS] = self.get_contained_ps(solution, must_contain=must_contain)

        print(f"The solution \n"
              f"{utils.indent(self.problem.repr_fs(solution.full_solution))}")

        explainability_coverage = self.get_explainability_percentage_of_solution(contained_pss)
        print(f"It is {int(explainability_coverage * 100)}% explainable")

        if len(contained_pss) > 0:
            print(f"contains the following PSs:")
        else:
            print("No matching PSs were found for the requested combination of solution and variable...")

        if shown_ps_max is None:
            shown_ps_max = 10000

        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps)))
            print()

    def explain_difference(self, solution_a: EvaluatedFS, solution_b: EvaluatedFS):
        print(f"Inspecting difference of solutions... "
              f"A.fitness = {solution_a.fitness}, "
              f"B.fitness = {solution_b.fitness}")

        self.lcs_manager.investigate_pair_if_necessary(solution_a, solution_b)

        in_a, in_b = self.get_difference_ps(solution_a, solution_b)

        def print_ps_and_descriptors(ps: PS):
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps)))
            print()

        print("In solution a we have")
        for ps in in_a:
            print_ps_and_descriptors(ps)

        print("\nIn solutions B we have")
        for ps in in_b:
            print_ps_and_descriptors(ps)

    def handle_solution_query(self, solutions: list[EvaluatedFS], ps_show_limit: int):
        index = int(input("Which solution? "))
        solution_to_explain = solutions[index]
        self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit)

    def handle_diff_query(self, solutions: list[EvaluatedFS]):
        index_a = int(input("Which solution is A?"))
        index_b = int(input("Which solution is B?"))
        solution_a = solutions[index_a]
        solution_b = solutions[index_b]

        self.explain_difference(solution_a, solution_b)

    def handle_variable_query(self):
        raise NotImplemented
        # variable_index = int(input("Which variable? "))
        # self.describe_properties_of_variable(variable_index)

    def handle_variable_within_solution_query(self, solutions: list[EvaluatedFS], ps_show_limit: int):
        raise NotImplemented
        # variable_index = int(input("Which variable? "))
        # solution_index = int(input("Which solution? "))
        # solution_to_explain = solutions[solution_index]
        # self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit, must_contain=variable_index)
        # self.describe_properties_of_variable(variable_index)

    def handle_plotvar_query(self):
        raise NotImplemented
        # variable_index = int(input("Which variable? "))
        # print("\tOptions for properties are "+", ".join(varname for varname in self.ps_property_manager.get_available_properties()))
        # property_name = input("Which property? ")
        #
        # self.ps_property_manager.plot_var_property(var_index=variable_index,
        #                                            value=None,
        #                                            property_name=property_name,
        #                                            pss=self.pss)

    def handle_global_query(self):
        raise NotImplemented
        # self.describe_global_information()

    def explanation_loop(self,
                         amount_of_fs_to_propose: int = 6,
                         ps_show_limit: int = 12,
                         show_debug_info=False):
        solutions = self.pRef.get_top_n_solutions(amount_of_fs_to_propose)

        print(f"The top {amount_of_fs_to_propose} solutions are")
        for solution in solutions:
            print(self.problem.repr_fs(solution.full_solution))
            print(f"(Has fitness {solution.fitness})")
            print()

        finish = False
        while not finish:
            answer = input("Type a command from [s, v, vs, plotvar, global], or n to exit: ")
            answer = answer.lower()
            try:
                if answer in {"s", "sol", "solution"}:
                    self.handle_solution_query(solutions, ps_show_limit)
                elif answer in {"d", "diff"}:
                    self.handle_diff_query(solutions)
                elif answer in {"v", "var", "variable"}:
                    self.handle_variable_query()
                elif answer in {"vs", "variable in solution"}:
                    self.handle_variable_within_solution_query(solutions, ps_show_limit)
                elif answer in {"pss", "ps", "partial solutions"}:
                    self.handle_pss_query()
                elif answer in {"pv", "plotvar"}:
                    self.handle_plotvar_query()
                elif answer in {"g", "global"}:
                    self.handle_global_query()
                elif answer in {"d", "distributions"}:
                    self.handle_distribution_query()
                elif answer in {"n", "no", "exit", "q", "quit"}:
                    finish = True
                else:
                    print(f"Sorry, the command {answer} was not recognised")
            except Exception as e:
                print(f"Something went wrong: {e}")
            finally:
                continue

        want_changes = input("Do you want the changes to be saved?")
        if want_changes.upper() == "Y":
            self.save_changes_and_quit()
        print("Bye Bye!")

    def get_ps_size_distribution(self):
        sizes = [ps.fixed_count() for ps in self.pss]
        unique_sizes = sorted(list(set(sizes)))

        def proportion_for_size(target_size: int) -> float:
            return len([1 for item in sizes if item == target_size]) / len(sizes)

        return {size: proportion_for_size(size)
                for size in unique_sizes}

    def handle_pss_query(self):
        raise NotImplemented

    def handle_distribution_query(self):
        print("PSs properties")
        self.problem.print_stats_of_pss(self.pss, self.pRef.get_top_n_solutions(50))

    def save_changes_and_quit(self):
        self.descriptors_manager.write_to_files()
