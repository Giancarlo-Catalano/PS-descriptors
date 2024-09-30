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

    lcs_environment: Optional[OneAtATimeSolutionDifferenceScenario]
    lcs_scenario: Optional[Scenario]

    algorithm: Optional[SolutionDifferenceAlgorithm]
    model: Optional[SolutionDifferenceModel]

    rule_conditions_file: str
    rule_attributes_file: str

    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 pRef: PRef,
                 rule_conditions_file: str,
                 rule_attributes_file: str,
                 verbose: bool = False):

        self.optimisation_problem = optimisation_problem
        self.pRef = pRef
        self.ps_evaluator = None
        self.lcs_environment = None
        self.lcs_scenario = None
        self.algorithm = None
        self.model = None

        self.rule_conditions_file = rule_conditions_file
        self.rule_attributes_file = rule_attributes_file

        self.verbose = verbose

    def load_from_existing_if_possible(self):
        conditions_file_exists = file_exists(self.rule_conditions_file)
        rule_attributes_file_exists = file_exists(self.rule_attributes_file)
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

            search_population = min(50, sum(self.optimisation_problem.search_space.cardinalities))

            self.ps_evaluator, self.lcs_environment, self.lcs_scenario, self.algorithm, self.model = self.get_objects_when_rules_are_unknown(
                optimisation_problem=self.optimisation_problem,
                pRef=self.pRef,
                covering_search_population=search_population,
                covering_search_budget=1000,
                training_cycles_per_solution=500,
                verbose=self.verbose)

    def load_from_files(self):
        pss = load_pss(self.rule_conditions_file)
        rules = self.get_rules_from_file(pss, self.rule_attributes_file, self.algorithm)
        self.model.set_rules(rules)

    def write_rules_to_files(self):
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
                                           pRef: PRef,
                                           covering_search_budget: int,
                                           covering_search_population: int,
                                           training_cycles_per_solution: int,
                                           verbose: bool = False):
        ps_evaluator = GeneralPSEvaluator(pRef)  # Evaluates Linkage and keeps track of PS evaluations used

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
                                                verbose_search=False)

        LCSManager.set_settings_for_lcs_algorithm(algorithm)

        # This should be a solutionDifferenceModel
        model = algorithm.new_model(scenario)
        model.verbose = verbose

        return ps_evaluator, lcs_environment, scenario, algorithm, model

    @classmethod
    def get_rules_from_file(cls,
                            pss: list[PS],
                            rule_attribute_file: str,
                            algorithm: SolutionDifferenceAlgorithm) -> list[xcs.XCSClassifierRule]:
        # internally it is a cvs file, where the columns are the fitness, error, experience, accuracy, correct_count,
        # time_stamp
        attribute_table = pd.read_csv(rule_attribute_file)

        def rule_from_row(ps: PS, row) -> xcs.XCSClassifierRule:
            rule = xcs.XCSClassifierRule(action=True,
                                         algorithm=algorithm,
                                         time_stamp=row["time_stamp"],
                                         condition=CombinatorialCondition.from_ps_values(ps.values))
            rule.fitness = row["fitness"]
            rule.error = row["error"]
            rule.experience = row["experience"]
            rule.time_stamp = row["time_stamp"]

            rule.accuracy = row["accuracy"]
            rule.correct_count = row["correct_count"]

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

    def explain_best_solution(self):
        found_optima = self.pRef.get_top_n_solutions(1)[0]
        print(f"The found optima is {self.optimisation_problem.repr_fs(found_optima)}")
        print(f"It has fitness {found_optima.fitness}")

        with announce("Training the model on the solution", self.verbose):
            matches = self.investigate_solution(found_optima)

            for rule in matches:
                print(self.optimisation_problem.repr_ps(rule.condition))
                print(f"Accuracy = {rule.accuracy:.2f}")

    def polish_on_entire_dataset(self):
        """NOTE this doesn't really work well"""
        entire_dataset_environment = RandomPairsScenario(original_problem=self.optimisation_problem,
                                                         pRef=self.pRef,
                                                         training_cycles=1000,
                                                         verbose=True)
        observer = ScenarioObserver(entire_dataset_environment)
        self.model.run(observer, learn=True)

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
        winner, loser = (solution_a, solution_b) if solution_a > solution_b else (solution_b, solution_a)
        if self.verbose:
            print(f"Comparing {winner} and {loser}")
        self.model.match(situation=(winner, loser))  # forces to cover if necessary

    # currently unused
    @classmethod
    def from_problem(cls,
                     optimisation_problem: BenchmarkProblem,
                     resolution_method: str,
                     pRef_size: int,
                     covering_search_budget: int,
                     covering_search_population: int,
                     training_cycles_per_solution: int,
                     flip_fitness: bool = False,
                     verbose: bool = False):
        # generate the reference population

        with announce(f"Generating the reference population using {resolution_method} of size {pRef_size}", verbose):
            pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                             sample_size=pRef_size,  # these are the Solution evaluations
                                             which_algorithm="uniform " + resolution_method,
                                             verbose=verbose)

        pRef = PRef.unique(pRef)

        if flip_fitness:
            pRef.fitness_array *= -1

        if verbose:
            print(f"After pruning the pRef, {pRef.sample_size} solutions are left")

        return cls.from_problem_and_pRef(optimisation_problem=optimisation_problem,
                                         covering_search_budget=covering_search_budget,
                                         covering_search_population=covering_search_population,
                                         pRef=pRef,
                                         training_cycles_per_solution=training_cycles_per_solution)

    # currently unused
    @classmethod
    def from_problem_and_rules(cls,
                               optimisation_problem: BenchmarkProblem,
                               pRef: PRef,
                               rules: list[xcs.XCSClassifierRule],
                               training_cycles_per_solution: int,
                               covering_search_budget: int,
                               covering_population_size: int,
                               verbose: bool = False):
        ps_evaluator = GeneralPSEvaluator(pRef)
        lcs_environment = OneAtATimeSolutionDifferenceScenario(original_problem=optimisation_problem,
                                                               pRef=pRef,  # where it gets the solutions from
                                                               training_cycles=training_cycles_per_solution,
                                                               # how many solutions to show (might repeat)
                                                               verbose=verbose)
        lcs_scenario = ScenarioObserver(lcs_environment)
        algorithm = SolutionDifferenceAlgorithm(ps_evaluator=ps_evaluator,
                                                xcs_problem=lcs_environment,
                                                covering_search_budget=covering_search_budget,
                                                covering_population_size=covering_population_size,
                                                verbose=verbose,
                                                verbose_search=verbose)

        LCSManager.set_settings_for_lcs_algorithm(algorithm)

        # This should be a solutionDifferenceModel
        model = algorithm.new_model_from_rules(lcs_scenario, rules)
        model.verbose = verbose

        return cls(optimisation_problem=optimisation_problem,
                   pRef=pRef,
                   ps_evaluator=ps_evaluator,
                   lcs_environment=lcs_environment,
                   lcs_scenario=lcs_scenario,
                   algorithm=algorithm,
                   model=model)

    def get_matches_with_partial_solution(self, partial_solution: PS) -> list[xcs.XCSClassifierRule]:
        return [rule
                for rule in self.get_rules_in_model()
                if rule.condition.matches_partial_solution(partial_solution)]


def test_human_in_the_loop_explainer():
    print("I am running")
    # optimisation_problem = RoyalRoad(4, 4)
    # optimisation_problem = Trapk(4, 5)
    optimisation_problem = EfficientBTProblem.from_default_files()
    covering_search_population = min(50, optimisation_problem.search_space.amount_of_parameters)
    amount_of_generations = 30
    explainer = LCSManager.from_problem(optimisation_problem=optimisation_problem,
                                        covering_search_budget=covering_search_population * amount_of_generations,
                                        covering_search_population=covering_search_population,
                                        pRef_size=10000,
                                        training_cycles_per_solution=100,
                                        resolution_method="GA",
                                        verbose=False)

    # explainer.explain_best_solution()
    explainer.explain_top_n_solutions(12)

# test_human_in_the_loop_explainer()
