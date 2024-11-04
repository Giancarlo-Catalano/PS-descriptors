import itertools
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import xcs
from tqdm import tqdm

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest
from Explanation.PRefManager import PRefManager
from LCS.Conversions import get_rules_in_model
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from LCS.DifferenceExplainer.PatchManager import PatchManager
from LCS.LCSManager import LCSManager
from LCS.PSEvaluator import GeneralPSEvaluator


class DifferenceExplainer:
    problem: BenchmarkProblem

    pRef_manager: PRefManager
    descriptors_manager: DescriptorsManager

    verbose: bool
    speciality_threshold: float

    ps_evaluator: GeneralPSEvaluator

    positive_lcs_manager: LCSManager
    negative_lcs_manager: LCSManager

    allow_positive_traits: bool
    allow_negative_traits: bool

    patch_manager: PatchManager

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 ps_files: (str, str, str),  # positive, control, negative
                 descriptors_file: str,
                 rule_attribute_files: (str, str),  # positive, negative
                 allow_positive_traits: bool,
                 allow_negative_traits: bool,
                 speciality_threshold: float,
                 verbose=False):
        positive_ps_file, control_ps_file, negative_ps_file = ps_files
        positive_rule_attribute_file, negative_rule_attribute_file = rule_attribute_files
        self.allow_negative_traits = allow_negative_traits
        self.allow_positive_traits = allow_positive_traits

        self.problem = problem
        self.pRef_manager = PRefManager(problem=problem,
                                        pRef_file=pRef_file,
                                        instantiate_own_evaluator=False,
                                        verbose=True)

        self.pRef_manager.load_from_existing_if_possible()

        self.descriptors_manager = DescriptorsManager(optimisation_problem=problem,
                                                      control_pss_file=control_ps_file,
                                                      control_descriptors_table_file=descriptors_file,
                                                      control_samples_per_size_category=1000,
                                                      pRef_manager=self.pRef_manager,
                                                      verbose=verbose)

        self.descriptors_manager.load_from_existing_if_possible()

        self.ps_evaluator = GeneralPSEvaluator(pRef=self.pRef_manager.pRef, optimisation_problem=problem)

        # the positive and negative manager are always initialised, but the flags allow_[neg, pos]_traits determine if they are used.

        self.positive_lcs_manager = LCSManager(optimisation_problem=problem,
                                               ps_evaluator=self.ps_evaluator,
                                               rule_conditions_file=positive_ps_file,
                                               rule_attributes_file=positive_rule_attribute_file,
                                               pRef=self.pRef,
                                               search_for_negative_traits=False,
                                               verbose=verbose)
        self.positive_lcs_manager.load_from_existing_if_possible()

        self.negative_lcs_manager = LCSManager(optimisation_problem=problem,
                                               ps_evaluator=self.ps_evaluator,
                                               rule_conditions_file=negative_ps_file,
                                               rule_attributes_file=negative_rule_attribute_file,
                                               pRef=self.pRef,
                                               search_for_negative_traits=True,
                                               verbose=verbose)
        self.negative_lcs_manager.load_from_existing_if_possible()

        self.patch_manager = PatchManager(lcs_manager=self.positive_lcs_manager,
                                          search_space=self.problem.search_space,
                                          merge_limit=100)  # TODO parametrise

        self.speciality_threshold = speciality_threshold
        self.verbose = verbose

    @classmethod
    def from_folder(cls,
                    problem: BenchmarkProblem,
                    folder: str,
                    allow_negative_traits: bool = False,
                    allow_positive_traits: bool = True,
                    speciality_threshold=0.1,
                    verbose=False):
        pRef_file = os.path.join(folder, "pRef.npz")
        positive_ps_file = os.path.join(folder, "positive_ps.npz")
        negative_ps_file = os.path.join(folder, "negative_ps.npz")
        control_ps_file = os.path.join(folder, "control_ps.npz")
        descriptors_file = os.path.join(folder, "descriptors.csv")
        positive_rule_attribute_file = os.path.join(folder, "positive_rule_attributes.csv")
        negative_rule_attribute_file = os.path.join(folder, "negative_rule_attributes.csv")

        return cls(
            problem=problem,
            pRef_file=pRef_file,
            ps_files=(positive_ps_file, control_ps_file, negative_ps_file),
            descriptors_file=descriptors_file,
            rule_attribute_files=(positive_rule_attribute_file, negative_rule_attribute_file),
            speciality_threshold=speciality_threshold,
            allow_negative_traits=allow_negative_traits,
            allow_positive_traits=allow_positive_traits,
            verbose=verbose)

    @property
    def positive_pss(self) -> list[PS]:
        return self.positive_lcs_manager.get_pss_from_model()

    @property
    def positive_rules(self) -> list[xcs.XCSClassifierRule]:
        return get_rules_in_model(self.positive_lcs_manager.model)

    @property
    def negative_pss(self) -> list[PS]:
        return self.negative_lcs_manager.get_pss_from_model()

    @property
    def negative_rules(self) -> list[xcs.XCSClassifierRule]:
        return get_rules_in_model(self.negative_lcs_manager.model)

    @property
    def all_pss(self) -> list[PS]:
        result = []
        if self.allow_positive_traits:
            result.extend(self.positive_pss)

        if self.allow_negative_traits:
            result.extend(self.negative_pss)

        return result

    @property
    def get_lcs_managers_in_use(self) -> list[LCSManager]:
        result = []
        if self.allow_positive_traits:
            result.append(self.positive_lcs_manager)
        if self.allow_negative_traits:
            result.append(self.negative_lcs_manager)

        return result

    @property
    def all_rules(self) -> list[xcs.XCSClassifierRule]:
        result = []
        if self.allow_positive_traits:
            result.extend(self.positive_rules)

        if self.allow_negative_traits:
            result.extend(self.negative_rules)

        return result

    @property
    def pRef(self) -> PRef:
        return self.pRef_manager.pRef

    def get_fitness_delta_string(self, ps: PS) -> str:
        p_value, _ = self.pRef_manager.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.pRef_manager.get_average_when_present_and_absent(ps)
        delta = avg_when_present - avg_when_absent

        return (f"({'POSITIVE' if delta >= 0 else 'NEGATIVE'}), "
                f"delta = {delta:.2f}, "
                f"avg when present = {avg_when_present:.2f}, "
                f"avg when absent = {avg_when_absent:.2f}")
        # f"p-value = {p_value:e}")



    def get_ps_description(self, ps: PS) -> str:
        return utils.indent("\n".join([self.problem.repr_extra_ps_info(ps),
                                       self.get_fitness_delta_string(ps),
                                       self.descriptors_manager.get_descriptors_string(ps)]))

    def get_difference_rules(self, solution_a: EvaluatedFS, solution_b: EvaluatedFS) -> (list[PS], list[PS]):
        in_a = []
        in_b = []
        for rule in self.all_rules:
            if rule.condition(solution_a):
                if not rule.condition(solution_b):
                    in_a.append(rule)
            elif rule.condition(solution_b):
                if not rule.condition(solution_a):
                    in_b.append(rule)

        return in_a, in_b

    def get_explainability_percentage_of_solution(self, rules: list[xcs.XCSClassifierRule]) -> float:
        if len(rules) == 0:
            return 0
        used_vars = np.array([rule.condition.values != STAR for rule in rules])
        used_vars_count = used_vars.sum(axis=0, dtype=bool).sum(dtype=int)
        size_of_solution = len(rules[0].condition)
        return used_vars_count / size_of_solution

    def print_ps_and_descriptors(self, ps: PS):
        print(self.problem.repr_ps(ps))
        print(utils.indent(self.get_ps_description(ps)))
        # print()

    def print_rule_and_descriptors(self, rule: xcs.XCSClassifierRule):
        print(self.problem.repr_ps(rule.condition))
        print(f"Accuracy = {int(rule.fitness * 100)}%, average error = {rule.error:.2f}, age = {rule.experience}")
        print(utils.indent(self.get_ps_description(rule.condition)))
        # print()

    def explain_difference(self, solution_a: EvaluatedFS, solution_b: EvaluatedFS):
        print(f"Inspecting difference of solutions... ")
        # f"A.fitness = {solution_a.fitness}, "
        # f"B.fitness = {solution_b.fitness}")

        self.ensure_pair_is_examined(solution_a, solution_b)

        in_a, in_b = self.get_difference_rules(solution_a, solution_b)

        if len(in_a) != 0:
            print("In solution A we have")
            for ps in in_a:
                self.print_rule_and_descriptors(ps)

        if len(in_b) != 0:
            print("\nIn solutions B we have")
            for ps in in_b:
                self.print_rule_and_descriptors(ps)

    def get_known_rules_in_solution(self, solution: EvaluatedFS) -> list[xcs.XCSClassifierRule]:
        return [rule for lcs_manager in self.get_lcs_managers_in_use
                for rule in lcs_manager.get_matches_with_solution(solution)]

    def ensure_pair_is_examined(self, sol_a: EvaluatedFS, sol_b: EvaluatedFS):
        if sol_a == sol_b:
            print("Warning: Attempted to check identical solutions")
            return
        for lcs_manager in self.get_lcs_managers_in_use:
            lcs_manager.investigate_pair_if_necessary(sol_a, sol_b)

    def explain_solution(self, solution: EvaluatedFS, amount_to_check_against: int):
        to_compare_against = PRefManager.get_most_similar_solutions_to(pRef=self.pRef, solution=solution,
                                                                       amount_to_return=amount_to_check_against)

        for alternative in to_compare_against:
            self.ensure_pair_is_examined(solution, alternative)

        rules_in_solution = self.get_known_rules_in_solution(solution)

        if len(rules_in_solution) == 0:
            print("No patterns were found in the solution")
            return

        coverage = self.get_explainability_percentage_of_solution(rules_in_solution)
        print(f"{len(rules_in_solution)} PSs were found, which cause coverage of {int(coverage * 100)}%")
        for rule in rules_in_solution:
            self.print_rule_and_descriptors(rule)

    def handle_solution_query(self, solutions: list[EvaluatedFS], ps_show_limit: int):
        index = int(input("Which solution? "))
        solution_to_explain = solutions[index]

        how_many_to_compare_against = int(input("How many solutions to compare it against?"))
        self.explain_solution(solution_to_explain, amount_to_check_against=how_many_to_compare_against)

    def handle_diff_query(self, solutions: list[EvaluatedFS]):
        index_a = int(input("Which solution is A? "))
        index_b = int(input("Which solution is B? "))
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
                         suppress_errors: bool = True):
        solutions = self.pRef.get_top_n_solutions(amount_of_fs_to_propose)

        print(f"The top {amount_of_fs_to_propose} solutions are")
        for solution in solutions:
            print(self.problem.repr_fs(solution))
            print(f"(Has fitness {solution.fitness})")
            print()

        while True:
            answer = input("Type a command from [s, diff, game], or n to exit: ")
            answer = answer.lower()

            def handle_answer() -> bool:
                wants_to_continue = True
                if answer in {"s", "sol", "solution"}:
                    self.handle_solution_query(solutions, ps_show_limit)
                elif answer in {"d", "diff"}:
                    self.handle_diff_query(solutions)
                elif len(answer) > 0 and answer[0] == "d":
                    parsed = utils.parse_simple_input(user_input=answer, format_string="d(X,X)")
                    if parsed is not None:
                        first_index, second_index = parsed
                        self.explain_difference(solutions[first_index], solutions[second_index])
                elif answer in {"game"}:
                    self.handle_game_query(solutions)
                elif answer in {"experiment"}:
                    self.replicable_experiments()
                elif answer in {"toggle-search-verbose", "toggle-verbose-search", "toggle_search_verbose",
                                "toggle_verbose_search"}:
                    for lcs_manager in self.get_lcs_managers_in_use:
                        lcs_manager.algorithm.toggle_verbose_search()
                    print(f"Verbose search was set to {self.positive_lcs_manager.algorithm.verbose_search}")
                elif answer in {"toggle_positive_traits", "toggle-positive-traits"}:
                    self.allow_positive_traits = not self.allow_positive_traits
                    print(f"The positive traits are set to {self.allow_negative_traits}")
                elif answer in {"toggle_negative_traits", "toggle-negative-traits"}:
                    self.allow_negative_traits = not self.allow_negative_traits
                    print(f"The negative traits are set to {self.allow_negative_traits}")
                elif answer in {"consistency_test"}:
                    solution_to_investigate = solutions[0]
                    solution_to_compare_against = PRefManager.get_most_similar_solution_to(self.pRef, solution_to_investigate)
                    self.rerun_explanation(solution_to_investigate,
                                           solution_to_compare_against,
                                           trials=100,
                                           for_positive=True)
                    self.rerun_explanation(solution_to_investigate,
                                           solution_to_compare_against,
                                           trials=100,
                                           for_positive=False)
                elif answer in {"rules"}:
                    self.handle_rules_query()
                elif answer in {"compare_delta_test"}:
                    self.delta_diff_experiment(solutions)
                elif answer in {"test_ps"}:
                    self.handle_test_ps_query()
                elif answer in {"v", "var", "variable"}:
                    self.handle_variable_query()
                elif answer in {"vs", "variable in solution"}:
                    self.handle_variable_within_solution_query(solutions, ps_show_limit)
                elif answer in {"pss", "ps", "partial solutions"}:
                    self.handle_pss_query()
                elif answer in {"n", "no", "exit", "q", "quit"}:
                    wants_to_continue = False
                else:
                    print(f"Sorry, the command {answer} was not recognised")

                return wants_to_continue

            if suppress_errors:
                try:
                    wants_to_continue = handle_answer()
                except Exception as e:
                    print(f"Something went wrong: {e}")
                finally:
                    continue
            else:
                wants_to_continue = handle_answer()

                if not wants_to_continue:
                    break

        want_changes = input("Do you want the changes to be saved? ")
        if want_changes.upper() == "Y":
            print("Saving changes")
            self.save_changes_and_quit()
        print("Bye Bye!")

    def handle_pss_query(self):
        for ps in self.all_pss:
            print(f"\t{ps}")

    def handle_rules_query(self):
        for rule in self.all_rules:
            print(rule)

    def save_changes_and_quit(self):
        self.positive_lcs_manager.write_rules_to_files()
        self.negative_lcs_manager.write_rules_to_files()
        self.descriptors_manager.write_to_files()

    def handle_game_query(self, solutions: list[EvaluatedFS]):
        index_of_solution_to_modify = int(input("Which solution to modify? "))
        solution_to_modify = solutions[index_of_solution_to_modify]
        quantity_of_variables_to_remove = int(input("How many variables to remove? "))

        incomplete_solution = self.patch_manager.remove_random_subset_from_solution(solution_to_modify,
                                                                                    quantity_of_variables_to_remove)

        print(f"The original solution was  {solution_to_modify}")
        print(f"The incomplete solution is {incomplete_solution}")

        by_method = defaultdict(list)
        methods = ["random", "P&M"]
        attempts_per_method = 3

        def complete_solution_via_method(method: str) -> EvaluatedFS:
            solution = self.patch_manager.fix_solution(incomplete_solution, method)
            fitness = self.problem.fitness_function(solution)
            return EvaluatedFS(solution, fitness)

        for method in methods:
            new_pss = [complete_solution_via_method(method)
                       for _ in range(attempts_per_method)]
            by_method[method].extend(new_pss)  # it's surprising that this works for a default dict

        all_finished_solutions = [solution
                                  for method in methods
                                  for solution in by_method[method]]

        random.shuffle(all_finished_solutions)

        print("The proposed solutions are: ")

        for index, solution in enumerate(all_finished_solutions):
            print(f"[{index}]\t{solution}")

        print("Type d(x, y) to compare x and y, or j(x) for just x, c(x) to make your final choice")

        while True:
            user_input = input("Select your action: ")

            if len(user_input) == 0:
                continue

            if user_input[0] == "d":
                parsed = utils.parse_simple_input(format_string="d(X, X)", user_input=user_input, explain_error=True)
                if parsed is None:
                    continue
                first_index, second_index = parsed
                solution_a, solution_b = all_finished_solutions[first_index], all_finished_solutions[second_index]
                self.explain_difference(solution_a, solution_b)
            elif user_input[0] == "j":
                parsed = utils.parse_simple_input(format_string="j(X)", user_input=user_input, explain_error=True)
                if parsed is None:
                    continue
                solution = all_finished_solutions[parsed[0]]

                self.explain_solution(solution, 12)
            elif user_input[0] == "c":
                parsed = utils.parse_simple_input(format_string="c(X)", user_input=user_input, explain_error=True)
                if parsed is None:
                    continue
                selected_solution = all_finished_solutions[parsed[0]]

                print("The results are:")
                for method in by_method:
                    print(f"{method = }")
                    for solution in by_method[method]:
                        index_in_list = all_finished_solutions.index(solution)
                        print(f"\t[{index_in_list}] -> {solution}")
                print("End of game")
                break  # freedom at last
            elif user_input in {"ls", "view"}:
                for index, solution in enumerate(all_finished_solutions):
                    print(f"[{index}]\t{solution}")
            else:
                print("The command was not recognised, please try again")

    def carry_out_experiment(self, seed: int,
                             amount_of_variables_to_remove: int,
                             solution_to_modify: EvaluatedFS,
                             alternatives_to_produce: int):
        random.seed(seed)
        incomplete_solution = self.patch_manager.remove_random_subset_from_solution(
            solution=solution_to_modify,
            amount_to_remove=amount_of_variables_to_remove)

        alternatives = [self.patch_manager.fix_solution(incomplete_solution, method="random")
                        for i in range(alternatives_to_produce)]
        alternatives = [EvaluatedFS(solution, self.problem.fitness_function(solution))
                        for solution in alternatives]

        all_selectable_solutions = [solution_to_modify] + alternatives

        def show_all_solutions():
            print(f"The original solution is:")
            print(f"\t[0]\t{solution_to_modify.as_full_solution()}")
            print("The other solutions are:")
            for index, alternative in enumerate(alternatives):
                print(f"\t[{index + 1}]\t{alternative.as_full_solution()}")

        def finish_game(selected_option: int):
            positions = ["1st", "2nd", "3rd"] + [f"{n}th" for n in range(4, 10)]
            best_fitness = max(alternative.fitness for alternative in alternatives)

            alternatives_with_index = list(enumerate(alternatives))
            alternatives_with_index.sort(key=utils.second, reverse=True)
            alternatives_with_index_and_position = [(alternative, original_index + 1, position)
                                                    for (position, (original_index, alternative)) in
                                                    enumerate(alternatives_with_index)]

            alternatives_with_index_and_position.sort(key=utils.second)

            for alternative, shown_index, position in alternatives_with_index_and_position:
                print(f"[{shown_index}]->{positions[position]}"
                      f"\t{alternative}, {'(your choice)' if shown_index == selected_option else ''}")

            your_position = alternatives_with_index_and_position[selected_option - 1][2]
            print(f"Your option was the {positions[your_position]}")
            if alternatives_with_index_and_position[selected_option - 1][0].fitness == best_fitness:
                print("Congrats!")

        print("The original solution is not usable, and your job is to find a suitable replacement")
        print(
            f"You will be shown {alternatives_to_produce} alternatives, and you should select the one with the highest fitness")
        show_all_solutions()

        print(f"You can type the following commands:"
              f"\n\td(X, Y) to view the explanation of the difference between solution X and Y"
              f"\n\tj(X) to view the patterns in solution X"
              f"\n\tc(X) to make your choice"
              f"\n\tview to view all the solutions again")

        while True:
            user_input = input("Select your action: ")

            if len(user_input) == 0:
                continue

            if user_input[0] == "d":
                parsed = utils.parse_simple_input(format_string="d(X,X)", user_input=user_input,
                                                  explain_error=True)
                if parsed is None:
                    continue
                self.explain_difference(all_selectable_solutions[parsed[0]], all_selectable_solutions[parsed[1]])
            elif user_input[0] == "j":
                parsed = utils.parse_simple_input(format_string="j(X)", user_input=user_input, explain_error=True)
                if parsed is None:
                    continue
                solution = all_selectable_solutions[parsed[0]]

                for alternative in alternatives:
                    self.ensure_pair_is_examined(solution, alternative)

                self.explain_solution(solution, 0)  # don't check against similar ones
            elif user_input[0] == "c":
                parsed = utils.parse_simple_input(format_string="c(X)", user_input=user_input, explain_error=True)
                if parsed is None:
                    continue

                if parsed[0] == 0 or parsed[0] >= len(all_selectable_solutions):
                    print("Please select a valid option")
                    continue

                finish_game(parsed[0])
                break  # freedom at last
            elif user_input in {"ls", "view"}:
                show_all_solutions()
            elif user_input in {"prepare"}:
                for sol_a, sol_b in itertools.combinations(all_selectable_solutions, r=2):
                    if sol_a == sol_b:
                        continue
                    self.ensure_pair_is_examined(sol_a, sol_b)
                    self.explain_difference(sol_a, sol_b)
            elif user_input in {"show_useful_output"}:
                for sol_a_index, sol_b_index in itertools.combinations(range(alternatives_to_produce + 1), r=2):
                    sol_a = all_selectable_solutions[sol_a_index]
                    sol_b = all_selectable_solutions[sol_b_index]

                    print(f"d({sol_a_index}, {sol_b_index})")
                    self.explain_difference(sol_a, sol_b)
            else:
                print("The command was not recognised, please try again")

    def replicable_experiments(self):
        print("Entering experiment mode...")
        solution_to_modify = self.pRef.get_top_n_solutions(1)[0]

        experiment_seeds = [6]

        for seed in experiment_seeds:
            self.carry_out_experiment(seed=seed,
                                      alternatives_to_produce=5,
                                      solution_to_modify=solution_to_modify,
                                      amount_of_variables_to_remove=6)

    def rerun_explanation(self,
                          sol_a: EvaluatedFS,
                          sol_b: EvaluatedFS,
                          trials: int,
                          for_positive=True):

        if sol_a.fitness == sol_b.fitness:
            raise Exception("Trying to get pss of a pair with identical fitness...")

        winner, loser = (sol_a, sol_b) if sol_a > sol_b else (sol_b, sol_a)

        def get_ps(for_positive=True) -> PS:
            algorithm = (self.positive_lcs_manager if for_positive else self.negative_lcs_manager).algorithm
            pss = algorithm.get_pss_for_pair(winner, loser, only_return_least_dependent=False, only_return_biggest=True)
            assert (len(pss) == 1)
            return pss[0]

        def get_hamming_distance(ps_a: PS, ps_b: PS) -> int:
            return np.sum(ps_a.values != ps_b.values)

        def get_jaccard_distance(ps_a: PS, ps_b: PS) -> float:
            fixed_a = ps_a.values != STAR
            fixed_b = ps_b.values != STAR
            intersection = np.sum(fixed_a & fixed_b)
            union = np.sum(fixed_a | fixed_b)

            return intersection / union

        def get_shared_workers(ps_a: PS, ps_b: PS) -> int:
            v_a, v_b = ps_a.values, ps_b.values
            return np.sum((v_a != STAR) & (v_b != STAR) & (v_a == v_b))

        pss = []
        for _ in tqdm(range(trials)):
            pss.append(get_ps(for_positive=for_positive))

        for ps in pss:
            print(ps)

        hamming_distances = [get_hamming_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        shared_workers = [get_shared_workers(a, b)
                          for a, b in itertools.combinations(pss, r=2)]

        jaccard_distances = [get_jaccard_distance(a, b)
                             for a, b in itertools.combinations(pss, r=2)]

        mean_shared_workers = np.average(shared_workers)
        std_shared_workers = np.std(shared_workers)

        print(f"{hamming_distances = }")
        print(f"{shared_workers = }")
        print(f"{jaccard_distances = }")

    def delta_diff_experiment(self, good_solutions: list[EvaluatedFS]):
        amount_of_trials = int(input("Amount of trials? "))

        other_solutions = [self.pRef.get_nth_solution(random.randrange(self.pRef.sample_size))
                           for good_sol in good_solutions]

        def select_different(group_a, group_b) -> (EvaluatedFS, EvaluatedFS):
            while True:
                a = random.choice(group_a)
                b = random.choice(group_b)
                if a.fitness != b.fitness:
                    if a < b:
                        a, b = b, a
                    return a, b

        def get_delta_of_rule(rule: xcs.XCSClassifierRule) -> float:
            return self.descriptors_manager.get_fitness_delta(rule.condition, self.pRef)

        def get_deltas_for_pair(solution_a, solution_b) -> (list[float], list[float]):
            self.ensure_pair_is_examined(solution_a, solution_b)

            in_a, in_b = self.get_difference_rules(solution_a, solution_b)

            return list(map(get_delta_of_rule, in_a)), list(map(get_delta_of_rule, in_b))

        def safe_average(deltas) -> Optional[float]:
            if len(deltas) == 0:
                return None
            else:
                return np.average(deltas)

        def trials_from_groups(kind: str, first_group, second_group):
            for _ in tqdm(range(amount_of_trials)):
                a, b = select_different(first_group, second_group)
                self.ensure_pair_is_examined(a, b)
                deltas_a, deltas_b = get_deltas_for_pair(a, b)

                print(f"{kind}\t{safe_average(deltas_a)}\t{safe_average(deltas_b)}\t{len(deltas_a)}\t{len(deltas_b)}")

        # good good
        trials_from_groups("GG", good_solutions, good_solutions)
        trials_from_groups("GO", good_solutions, other_solutions)
        trials_from_groups("OO", other_solutions, other_solutions)

    def handle_test_ps_query(self):
        pss = self.all_pss

        for index, ps, in enumerate(pss):
            print(f"[{index}]\t{ps}")

        selection = int(input("Please select the ps to be tested "))
        if selection < 0 or selection > len(pss):
            raise Exception("Invalid index entered for pss")

        selected_ps = pss[selection]
        fs_evaluator = FSEvaluator(self.problem.fitness_function)
        tester = WilcoxonTest(sample_size=100,
                              search_space=self.problem.search_space,
                              fitness_evaluator=fs_evaluator)

        beneficial_p_value, maleficial_p_value = tester.get_p_values_of_ps(selected_ps)

        print(f"The beneficial p-value is {beneficial_p_value:.4f}")
        print(f"The maleficial p-value is {maleficial_p_value:.4f}")
