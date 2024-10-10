import csv
import random
from typing import Literal, Optional

import numpy as np
import pandas as pd
import xcs
from pandas.io.common import file_exists
from xcs.scenarios import Scenario, ScenarioObserver

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from LCS.Conversions import get_rules_in_action_set, get_rules_in_model
from LCS.PSEvaluator import GeneralPSEvaluator
from LCS.XCSComponents.CombinatorialRules import CombinatorialCondition
from LCS.XCSComponents.SolutionDifferenceAlgorithm import SolutionDifferenceAlgorithm
from LCS.XCSComponents.SolutionDifferenceModel import SolutionDifferenceModel
from LCS.XCSComponents.SolutionDifferenceScenario import OneAtATimeSolutionDifferenceScenario, RandomPairsScenario
from PSMiners.Mining import load_pss, write_pss_to_file
from utils import announce


class LCSManager:
    optimisation_problem: BenchmarkProblem
    pRef: PRef
    ps_evaluator: Optional[GeneralPSEvaluator]

    algorithm: Optional[SolutionDifferenceAlgorithm]
    model: Optional[SolutionDifferenceModel]

    rule_conditions_file: str
    rule_attributes_file: str

    search_for_negative_traits: bool
    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 pRef: PRef,
                 ps_evaluator: GeneralPSEvaluator,
                 rule_conditions_file: str,
                 rule_attributes_file: str,
                 search_for_negative_traits: bool = False,
                 verbose: bool = False):

        self.optimisation_problem = optimisation_problem
        self.pRef = pRef
        self.ps_evaluator = ps_evaluator
        self.lcs_environment = None
        self.lcs_scenario = None
        self.algorithm = None
        self.model = None

        self.rule_conditions_file = rule_conditions_file
        self.rule_attributes_file = rule_attributes_file

        self.search_for_negative_traits = search_for_negative_traits

        self.verbose = verbose

    def load_from_existing_if_possible(self):
        conditions_file_exists = file_exists(self.rule_conditions_file)
        rule_attributes_file_exists = file_exists(self.rule_attributes_file)

        search_population = min(100, sum(self.optimisation_problem.search_space.cardinalities))

        self.lcs_environment, self.lcs_scenario, self.algorithm, self.model = self.get_objects_when_rules_are_unknown(
            ps_evaluator=self.ps_evaluator,
            optimisation_problem=self.optimisation_problem,
            pRef=self.pRef,
            covering_search_population=search_population,
            covering_search_budget=1000,
            training_cycles_per_solution=500,
            search_for_negative_traits=self.search_for_negative_traits,
            verbose=self.verbose)


        if conditions_file_exists and rule_attributes_file_exists:
            if self.verbose:
                print(
                    f"Found a pre-calculated LCS, loading from {self.rule_conditions_file} and {self.rule_attributes_file}")
            self.load_from_files()
        else:
            if conditions_file_exists != rule_attributes_file_exists:
                raise Exception("Only one of the files for the control data is present!")

            if self.verbose:
                print(f"Since no LCS files were found, the LCS model will be initialised as empty")



    def load_from_files(self):
        pss = load_pss(self.rule_conditions_file)
        rules = self.get_rules_from_file(pss, self.rule_attributes_file, self.algorithm)
        self.model.set_rules(rules)

    def write_rules_to_files(self):
        if self.verbose:
            print(f"Writing ({'negative' if self.search_for_negative_traits else 'positive'} traits to files {self.rule_conditions_file}, {self.rule_attributes_file}")
        write_pss_to_file(self.get_pss_from_model(), self.rule_conditions_file)
        self.write_rule_attributes_to_file(get_rules_in_model(self.model), self.rule_attributes_file)

    @classmethod
    def set_settings_for_lcs_algorithm(cls, algorithm: xcs.XCSAlgorithm) -> None:
        """Simply sets the settings that are best for my purposes"""
        # play with these settings ad lib.
        algorithm.crossover_probability = 0
        algorithm.deletion_threshold = 50  # minimum age before a rule can be pruned away
        # algorithm.discount_factor = 0
        algorithm.do_action_set_subsumption = True
        # algorithm.do_ga_subsumption = True
        # algorithm.exploration_probability = 0
        # algorithm.ga_threshold = 100000
        algorithm.max_population_size = 100
        # algorithm.exploration_probability = 0
        # algorithm.minimum_actions = 1
        algorithm.subsumption_threshold = 1  # minimum age before a rule can subsume another

        algorithm.allow_ga_reproduction = False

    @classmethod
    def get_objects_when_rules_are_unknown(cls,
                                           optimisation_problem: BenchmarkProblem,
                                           ps_evaluator: GeneralPSEvaluator,
                                           pRef: PRef,
                                           covering_search_budget: int,
                                           covering_search_population: int,
                                           training_cycles_per_solution: int,
                                           search_for_negative_traits: bool,
                                           verbose: bool = False):

        lcs_environment = OneAtATimeSolutionDifferenceScenario(original_problem=optimisation_problem,
                                                               pRef=pRef,  # where it gets the solutions from
                                                               training_cycles=training_cycles_per_solution,
                                                               # how many solutions to show (might repeat)
                                                               verbose=verbose)

        scenario = ScenarioObserver(lcs_environment)

        # my custom XCS algorithm
        algorithm = SolutionDifferenceAlgorithm(ps_evaluator=ps_evaluator,
                                                xcs_problem=lcs_environment,
                                                covering_search_budget=covering_search_budget,
                                                covering_population_size=covering_search_population,
                                                verbose=verbose,
                                                search_for_negative_traits = search_for_negative_traits,
                                                verbose_search=False)

        LCSManager.set_settings_for_lcs_algorithm(algorithm)

        # This should be a solutionDifferenceModel
        model = algorithm.new_model(scenario)
        model.verbose = verbose

        return lcs_environment, scenario, algorithm, model

    @classmethod
    def get_rules_from_file(cls,
                            pss: list[PS],
                            rule_attribute_file: str,
                            algorithm: SolutionDifferenceAlgorithm) -> list[xcs.XCSClassifierRule]:
        # internally it is a cvs file, where the columns are the fitness, error, experience, accuracy, correct_count,
        # time_stamp
        attribute_table = pd.read_csv(rule_attribute_file)

        def rule_from_row(ps: PS, row) -> xcs.XCSClassifierRule:
            row_dict = dict(row[1])
            rule = xcs.XCSClassifierRule(action=True,
                                         algorithm=algorithm,
                                         time_stamp=int(row_dict["time_stamp"]),
                                         condition=CombinatorialCondition.from_ps_values(ps.values))
            rule.fitness = float(row_dict["fitness"])
            rule.error = float(row_dict["error"])
            rule.experience = int(row_dict["experience"])
            rule.time_stamp = float(row_dict["time_stamp"])

            rule.accuracy = float(row_dict["accuracy"])
            rule.correct_count = int(row_dict["correct_count"])

            return rule

        return [rule_from_row(ps, row) for ps, row in zip(pss, attribute_table.iterrows())]

    def get_pss_from_model(self) -> list[PS]:
        return [rule.condition for rule in get_rules_in_model(self.model)]

    @classmethod
    def write_rule_attributes_to_file(self, rules: list[xcs.XCSClassifierRule], file_location: str):
        headers = ["fitness", "error", "experience", "time_stamp", "accuracy", "correct_count"]
        with open(file_location, mode="w") as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(headers)

            for rule in rules:
                rule_row = [rule.fitness, rule.error, rule.experience, rule.time_stamp,
                            rule.accuracy if hasattr(rule, "accuracy") else 0,
                            rule.correct_count if hasattr(rule, "correct_count") else 0]
                writer.writerow(rule_row)

    def investigate_solution(self, solution: EvaluatedFS) -> list[xcs.ClassifierRule]:
        self.lcs_environment.set_solution_to_investigate(solution)
        self.model.run(self.lcs_scenario, learn=True)

        return self.get_matches_with_solution(solution)

    def get_rules_in_model(self) -> list[xcs.XCSClassifierRule]:
        return [item[True] for item in self.model._population.values()]

    def get_matches_with_solution(self, solution: EvaluatedFS) -> list[xcs.XCSClassifierRule]:
        return [rule for rule in self.get_rules_in_model()
                if rule.condition(solution)]

    def get_matches_for_pair(self,
                             winner: EvaluatedFS,
                             loser: EvaluatedFS) -> (list[xcs.ClassifierRule], list[xcs.ClassifierRule]):
        match_set = self.model.match(situation=(winner, loser))

        correct_action_set, wrong_action_set = match_set[True], match_set[False]

        return get_rules_in_action_set(correct_action_set), get_rules_in_action_set(wrong_action_set)



    def explain_top_n_solutions(self, n: int):
        def print_model():
            rules: list[xcs.XCSClassifierRule] = get_rules_in_model(self.model)
            rules.sort(key=lambda x: x.accuracy, reverse=True)
            for rule in rules:
                print(self.optimisation_problem.repr_ps(rule.condition), end="")
                print(f"\t acc={rule.fitness:.2f}, error={rule.error:.2f}, age={rule.experience:.2f}\n")

        def generate_data():

            def random_good_solution() -> EvaluatedFS:
                return random.choice(solutions_to_explain)

            def random_solution() -> EvaluatedFS:
                return self.pRef.get_random_evaluated_fs()

            samples_to_collect = 100

            def check_pair(first: EvaluatedFS, second: EvaluatedFS) -> (int, int, float, float):
                winner, loser = (first, second) if first > second else (second, first)
                correct, wrong = self.get_matches_for_pair(winner, loser)
                correct_average_accuracy = 0 if len(correct) == 0 else np.average([rule.fitness for rule in correct])
                wrong_average_accuracy = 0 if len(wrong) == 0 else np.average([rule.fitness for rule in wrong])
                return len(correct), len(wrong), correct_average_accuracy, wrong_average_accuracy

            def generate_pair(how: Literal["both_good", "both_any", "one_good"]) -> (EvaluatedFS, EvaluatedFS):
                def pair_has_different_fitnesses(pair):
                    return pair[0].fitness != pair[1].fitness

                def generate_unsafe_pair() -> (EvaluatedFS, EvaluatedFS):
                    if how == "both_good":
                        return random_good_solution(), random_good_solution()
                    elif how == "both_any":
                        return random_solution(), random_solution()
                    else:
                        return random_good_solution(), random_solution()

                pair = generate_unsafe_pair()
                while not pair_has_different_fitnesses(pair):
                    pair = generate_unsafe_pair()
                return pair

            results = dict()
            for how in ["both_good", "both_any", "one_good"]:
                results[how] = []
                for iteration in range(samples_to_collect):
                    first, second = generate_pair(how)
                    result_pair = check_pair(first, second)
                    results[how].append(result_pair)

            def pretty_print_results(results_dict: dict):
                for pair_kind in results_dict:
                    for row in results_dict[pair_kind]:
                        # count_correct, count_wrong, accuracy_correct, accuracy_wrong = results_dict[pair_kind]
                        print("\t".join(f"{x}" for x in [pair_kind] + list(row)))

            pretty_print_results(results)

        with announce(f"Inspecting the best solutions"):
            solutions_to_explain = self.pRef.get_top_n_solutions(n)

        for solution in solutions_to_explain:
            self.investigate_solution(solution)

        print("At the end of the investigation, the model is")
        print_model()

        # print("Now polishing on the entire dataset")
        # self.polish_on_entire_dataset()
        #
        # print("After polishing, the model is ")
        # print_model()

    def investigate_pair_if_necessary(self, solution_a: EvaluatedFS, solution_b: EvaluatedFS):
        if solution_a == solution_b:
            print("Warning: You requested to check a solution against itself.")
            return
        winner, loser = (solution_a, solution_b) if solution_a > solution_b else (solution_b, solution_a)
        # if self.verbose:
        #     print(f"Comparing {winner} and {loser}")
        match_set = self.model.match(situation=(winner, loser))  # forces to cover if necessary

        self.model.use_match_set_for_learning(match_set)

    def get_matches_with_partial_solution(self, partial_solution: PS) -> list[xcs.XCSClassifierRule]:
        return [rule
                for rule in self.get_rules_in_model()
                if rule.condition.matches_partial_solution(partial_solution)]


