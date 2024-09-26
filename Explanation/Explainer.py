import os
from typing import Optional

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Core.PSMetric.Linkage.Additivity import MutualInformation
from Explanation.MinedPSManager import MinedPSManager
from Explanation.MutualInformationManager import MutualInformationManager
from Explanation.PRefManager import PRefManager
from Explanation.PSPropertyManager import PSPropertyManager


class Explainer:
    problem: BenchmarkProblem

    pRef_manager: PRefManager   # manages a npz
    mined_ps_manager: MinedPSManager  # manages some npz files
    ps_property_manager: PSPropertyManager   # which will manage a csv
    mutual_information_manager: MutualInformationManager # manages an npz

    minimum_acceptable_ps_size: int
    verbose: bool

    speciality_threshold: float

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 ps_file: str,
                 control_ps_file: str,
                 properties_file: str,
                 mutual_information_linkage_table_file: str,
                 speciality_threshold: float,
                 minimum_acceptable_ps_size: int = 2,
                 verbose = False):
        self.problem = problem
        self.pRef_manager = PRefManager(problem = problem,
                                        pRef_file = pRef_file,
                                        verbose=True)
        self.mined_ps_manager = MinedPSManager(problem = problem,
                                               mined_ps_file=ps_file,
                                               control_ps_file=control_ps_file,
                                               verbose=verbose)
        self.ps_property_manager = PSPropertyManager(problem = problem,
                                                     property_table_file=properties_file,
                                                     verbose=verbose,
                                                     threshold=speciality_threshold)

        self.mutual_information_manager = MutualInformationManager(linkage_table_file=mutual_information_linkage_table_file,
                                                                   cached_mutual_information=None,
                                                                   verbose=verbose)

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
        mutual_information_linkage_file = os.path.join(folder, "linkage_table.npz")

        return cls(problem = problem,
                   pRef_file = pRef_file,
                   ps_file = ps_file,
                   control_ps_file = control_ps_file,
                   properties_file = properties_file,
                   speciality_threshold = speciality_threshold,
                   mutual_information_linkage_table_file = mutual_information_linkage_file,
                   verbose=verbose)


    @property
    def pss(self) -> list[EvaluatedPS]:
        return self.mined_ps_manager.pss


    @property
    def pRef(self) -> PRef:
        return self.pRef_manager.pRef

    @property
    def mutual_information_metric(self) -> MutualInformation:
        return self.mutual_information_manager.mutual_information_metric



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
        #return sort_by_influence(pss, self.pRef)

        def get_atomicity(ps: EvaluatedPS) -> float:
            return ps.metric_scores[2]

        def get_mean_fitness(ps: EvaluatedPS) -> float:
            return ps.metric_scores[1]

        def get_simplicity(ps: EvaluatedPS) -> float:
            return ps.metric_scores[0]

        return utils.sort_by_combination_of(pss, key_functions=[get_mean_fitness, get_atomicity], reverse=False)


    def get_explainability_percentage_of_solution(self, pss: list[PS]) -> float:
        if len(pss) == 0:
            return 0
        used_vars = np.array([ps.values != STAR for ps in pss])
        used_vars_count = used_vars.sum(axis=0, dtype=bool).sum(dtype=int)
        return used_vars_count / len(pss[0])

    def explain_solution(self, solution: EvaluatedFS, shown_ps_max: int, must_contain: Optional[int] = None):
        contained_pss: list[EvaluatedPS] = self.get_contained_ps(solution, must_contain = must_contain)
        contained_pss = self.sort_pss(contained_pss)

        print(f"The solution \n"
              f"{utils.indent(self.problem.repr_fs(solution.full_solution))}")

        explainability_coverage = self.get_explainability_percentage_of_solution(contained_pss)
        print(f"It is {int(explainability_coverage*100)}% explainable")

        if len(contained_pss) > 0:
            print(f"contains the following PSs:")
        else:
            print("No matching PSs were found for the requested combination of solution and variable...")

        for ps in contained_pss[:shown_ps_max]:
            print(self.problem.repr_ps(ps))
            print(utils.indent(self.get_ps_description(ps)))
            print()



    def handle_solution_query(self, solutions: list[EvaluatedFS], ps_show_limit: int):
        index = int(input("Which solution? "))
        solution_to_explain = solutions[index]
        self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit)


    def handle_variable_query(self):
        variable_index = int(input("Which variable? "))
        self.describe_properties_of_variable(variable_index)

    def handle_variable_within_solution_query(self, solutions: list[EvaluatedFS], ps_show_limit: int):
        variable_index = int(input("Which variable? "))
        solution_index = int(input("Which solution? "))
        solution_to_explain = solutions[solution_index]
        self.explain_solution(solution_to_explain, shown_ps_max=ps_show_limit, must_contain = variable_index)
        self.describe_properties_of_variable(variable_index)

    def handle_plotvar_query(self):
        variable_index = int(input("Which variable? "))
        print("\tOptions for properties are "+", ".join(varname for varname in self.ps_property_manager.get_available_properties()))
        property_name = input("Which property? ")

        self.ps_property_manager.plot_var_property(var_index=variable_index,
                                                   value=None,
                                                   property_name=property_name,
                                                   pss=self.pss)



    def handle_global_query(self):
        self.describe_global_information()


    def explanation_loop(self,
                         amount_of_fs_to_propose: int = 6,
                         ps_show_limit: int = 12,
                         show_debug_info = False):
        solutions = self.get_best_n_full_solutions(amount_of_fs_to_propose)

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
                #TODO convert this into a match statement
                if answer in {"s", "sol", "solution"}:
                    self.handle_solution_query(solutions, ps_show_limit)
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

        print("Bye Bye!")



    def generate_files_with_default_settings(self,
                                             pRef_size: Optional[int] = 10000,
                                             pss_budget: Optional[int] = 10000,
                                             force_include_in_pRef: Optional[list[FullSolution]] = None):

        self.pRef_manager.generate_pRef_file(sample_size=pRef_size,
                                             which_algorithm="uniform SA",
                                             force_include=force_include_in_pRef)

        self.mined_ps_manager.generate_ps_file(pRef = self.pRef,
                                               population_size=50,
                                               ps_budget_in_total=pss_budget,
                                               ps_budget_per_run=1000)
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

        # print("The problem specific global information is")
        # self.problem.get_problem_specific_global_information(self.get_best_n_full_solutions(10))

    def describe_properties_of_variable(self, var: int, value: Optional[int] = None):

        print(f"Significant properties for the variable {var}:"+("" if value is None else f"when it's  = {value}"))

        property_stats = self.ps_property_manager.get_variable_properties_stats(self.pss, var, value)
        properties = [(prop, p_value, prop_mean, control_mean)
                      for prop, (p_value, prop_mean, control_mean) in property_stats.items()]
        properties.sort(key=utils.second)

        for prop, p_value, prop_mean, control_mean in properties:
            if p_value < 0.05:
                comparison_str = "lower" if  prop_mean < control_mean else "higher"
                print(f"* \t{self.problem.get_readable_property_name(prop)} is {comparison_str} than average,"
                      f"\n\t\t with p-value {p_value:.5f}, prop_mean = {prop_mean:.2f}, control_mean = {control_mean:.2f}")

    def handle_pss_query(self):
        pss = self.sort_pss(self.pss)
        for ps in pss:
            if isinstance(ps, EvaluatedPS):
                s, m, a = ps.metric_scores
                print(f"{ps}, "
                      f"\n\t{s=:.2f},"
                      f"\n\t{m=:.2f},"
                      f"\n\t{a=:.4f},"
                      f"")
            else:
                print(f"{ps}")

    def handle_distribution_query(self):
        print("PSs properties")
        self.problem.print_stats_of_pss(self.pss, self.get_best_n_full_solutions(50))






















