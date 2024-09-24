import heapq
import random
from typing import Literal

import numpy as np
import xcs
from xcs.scenarios import Scenario, ScenarioObserver

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from LCS.Conversions import get_rules_in_action_set
from LCS.SolutionDifference.SolutionDifferenceAlgorithm import SolutionDifferenceAlgorithm
from LCS.SolutionDifference.SolutionDifferenceModel import SolutionDifferenceModel
from LCS.SolutionDifference.SolutionDifferenceScenario import SolutionDifferenceScenario, \
    OneAtATimeSolutionDifferenceScenario
from LightweightLocalPSMiner.TwoMetrics import GeneralPSEvaluator
from utils import announce


class HumanInTheLoopExplainer:
    optimisation_problem: BenchmarkProblem
    pRef: PRef
    ps_evaluator: GeneralPSEvaluator

    lcs_environment: OneAtATimeSolutionDifferenceScenario
    lcs_scenario: Scenario

    algorithm: SolutionDifferenceAlgorithm
    model: SolutionDifferenceModel

    verbose: bool

    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 pRef: PRef,
                 ps_evaluator: GeneralPSEvaluator,
                 lcs_environment: OneAtATimeSolutionDifferenceScenario,
                 lcs_scenario: Scenario,
                 algorithm: SolutionDifferenceAlgorithm,
                 model: SolutionDifferenceModel,
                 verbose: bool = False):
        self.optimisation_problem = optimisation_problem
        self.pRef = pRef
        self.ps_evaluator = ps_evaluator
        self.lcs_environment = lcs_environment
        self.lcs_scenario = lcs_scenario
        self.algorithm = algorithm
        self.model = model

        self.verbose = verbose

    @classmethod
    def set_settings_for_lcs_algorithm(cls, algorithm: xcs.XCSAlgorithm) -> None:
        """Simply sets the settings that are best for my purposes"""
        # play with these settings ad lib. Leaving the defaults seems to work :--)
        algorithm.crossover_probability = 0
        algorithm.deletion_threshold = 10  # minimum age before a rule can be pruned away
        # algorithm.discount_factor = 0
        algorithm.do_action_set_subsumption = True
        # algorithm.do_ga_subsumption = True
        # algorithm.exploration_probability = 0
        # algorithm.ga_threshold = 100000
        algorithm.max_population_size = 100
        # algorithm.exploration_probability = 0
        # algorithm.minimum_actions = 1
        algorithm.subsumption_threshold = 10  # minimum age before a rule can subsume another

        algorithm.allow_ga_reproduction = False

    @classmethod
    def from_problem(cls,
                     optimisation_problem: BenchmarkProblem,
                     resolution_method: str,
                     pRef_size: int,
                     covering_search_budget: int,
                     covering_search_population: int,
                     training_cycles_per_solution: int,
                     verbose: bool = False):
        # generate the reference population

        with announce(f"Generating the reference population using {resolution_method} of size {pRef_size}", verbose):
            pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                             sample_size=pRef_size,  # these are the Solution evaluations
                                             which_algorithm="uniform " + resolution_method,
                                             verbose=verbose)

        pRef = PRef.unique(pRef)
        if verbose:
            print(f"After pruning the pRef, {pRef.sample_size} solutions are left")

        # generate the other components of the LCS system
        ps_evaluator = GeneralPSEvaluator(pRef)  # Evaluates Linkage and keeps track of PS evaluations used

        xcs_problem = OneAtATimeSolutionDifferenceScenario(original_problem=optimisation_problem,
                                                           pRef=pRef,  # where it gets the solutions from
                                                           training_cycles=training_cycles_per_solution,
                                                           # how many solutions to show (might repeat)
                                                           verbose=verbose)

        scenario = ScenarioObserver(xcs_problem)

        # my custom XCS algorithm, which just overrides when covering is required, and how it happens
        algorithm = SolutionDifferenceAlgorithm(ps_evaluator=ps_evaluator,
                                                xcs_problem=xcs_problem,
                                                covering_search_budget=covering_search_budget,
                                                covering_population_size=covering_search_population,
                                                verbose=verbose,
                                                verbose_search=verbose)

        HumanInTheLoopExplainer.set_settings_for_lcs_algorithm(algorithm)

        # This should be a solutionDifferenceModel
        model = algorithm.new_model(scenario)
        model.verbose = verbose

        return cls(optimisation_problem=optimisation_problem,
                   pRef=pRef,
                   ps_evaluator=ps_evaluator,
                   lcs_environment=xcs_problem,
                   lcs_scenario=scenario,
                   algorithm=algorithm,
                   model=model,
                   verbose = verbose)

    def investigate_solution(self, solution: EvaluatedFS) -> list[xcs.ClassifierRule]:
        self.lcs_environment.set_solution_to_investigate(solution)
        self.model.run(self.lcs_scenario, learn=True)

        return self.get_matches_with_solution(solution)

    def get_rules_in_model(self) -> list[xcs.ClassifierRule]:
        return [item[True] for item in self.model._population.values()]

    def get_matches_with_solution(self, solution: EvaluatedFS) -> list[xcs.ClassifierRule]:
        return [rule for rule in self.get_rules_in_model()
                if rule.condition(solution)]


    def get_matches_for_pair(self,
                             winner: EvaluatedFS,
                             loser: EvaluatedFS) -> (list[xcs.ClassifierRule], list[xcs.ClassifierRule]):
        match_set = self.model.match(situation = (winner, loser))

        correct_action_set, wrong_action_set = match_set[True], match_set[False]

        return get_rules_in_action_set(correct_action_set), get_rules_in_action_set(wrong_action_set)


    def get_n_best_solutions(self, n: int) -> list[EvaluatedFS]:
        indexes_and_fitnesses = list(enumerate(self.pRef.fitness_array))
        best_indexes_and_fitnesses = heapq.nlargest(n=n, iterable=indexes_and_fitnesses, key=utils.second)
        def get_nth_solution(index: int, fitness: float) -> EvaluatedFS:
            return EvaluatedFS(FullSolution(self.pRef.full_solution_matrix[index]),
                               fitness=fitness)
        return [get_nth_solution(index, fitness) for index, fitness in best_indexes_and_fitnesses]



    def explain_best_solution(self):
        found_optima = self.get_n_best_solutions(1)[0]
        print(f"The found optima is {self.optimisation_problem.repr_fs(found_optima)}")
        print(f"It has fitness {found_optima.fitness}")

        with announce("Training the model on the solution", self.verbose):
            matches = self.investigate_solution(found_optima)

            for rule in matches:
                print(self.optimisation_problem.repr_ps(rule.condition))
                print(f"Accuracy = {rule.accuracy:.2f}")


    def explain_top_n_solutions(self, n: int):
        solutions_to_explain = self.get_n_best_solutions(n)
        for solution in solutions_to_explain:
            self.investigate_solution(solution)

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
                    #count_correct, count_wrong, accuracy_correct, accuracy_wrong = results_dict[pair_kind]
                    print("\t".join(f"{x}" for x in [pair_kind]+list(row)))

        pretty_print_results(results)




def test_human_in_the_loop_explainer():
    #optimisation_problem = RoyalRoad(4, 4)
    optimisation_problem = Trapk(4, 5)
    #optimisation_problem = EfficientBTProblem.from_default_files()
    covering_search_population = min(50, optimisation_problem.search_space.amount_of_parameters)
    amount_of_generations = 30
    explainer = HumanInTheLoopExplainer.from_problem(optimisation_problem=optimisation_problem,
                                                     covering_search_budget=covering_search_population * amount_of_generations,
                                                     covering_search_population=covering_search_population,
                                                     pRef_size=10000,
                                                     training_cycles_per_solution=500,
                                                     resolution_method="GA",
                                                     verbose=True)

    #explainer.explain_best_solution()
    explainer.explain_top_n_solutions(4)


test_human_in_the_loop_explainer()

