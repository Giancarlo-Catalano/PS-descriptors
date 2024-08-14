import random
from collections import defaultdict
from typing import Literal, TypeAlias

import numpy as np

import utils
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from OwnLCS.Rule import Rule
from OwnLCS.RuleManager import RuleManager
from utils import get_count_report

OutputClass: TypeAlias = int


class OnlineLearner:
    solutions_and_fitnesses: list[EvaluatedFS]
    rule_manager: RuleManager

    tournaments_per_new_solution: int

    def __init__(self,
                 solutions_and_fitnesses: list[EvaluatedFS],
                 rule_manager: RuleManager,
                 tournaments_per_new_solution: int):
        self.solutions_and_fitnesses = solutions_and_fitnesses
        self.rule_manager = rule_manager
        self.tournaments_per_new_solution = tournaments_per_new_solution

    def pass_new_solution(self, solution: EvaluatedFS):
        competitors = random.choices(self.solutions_and_fitnesses, k=self.tournaments_per_new_solution)
        outcomes = [int(solution.fitness > competitor.fitness) for competitor in competitors]

        training_instances = []
        for competitor, outcome in zip(competitors, outcomes):
            training_instances.append((solution, outcome))
            training_instances.append((competitor, 1 - outcome))

        self.rule_manager.apply_training_batch(training_instances)


    def guess_class_naive(self, solution: FullSolution) -> OutputClass:
        matching_rules = self.rule_manager.get_matching_rules(solution)
        guessed_classes = [rule.predicted_class for rule in matching_rules]
        action_counts = get_count_report(guessed_classes)
        return max(action_counts.items(), key=utils.second)[0]


    def guess_class_weighted_on_numerosity_and_accuracy(self, solution: FullSolution) -> OutputClass:
        matching_rules = self.rule_manager.get_matching_rules(solution)
        def get_rule_weight(rule:Rule) -> float:
            return rule.numerosity * rule.get_accuracy()

        action_counts = defaultdict(float)
        for rule in matching_rules:
            action_counts[rule.predicted_class] = action_counts[rule.predicted_class] + get_rule_weight(rule)

        return max(action_counts.items(), key=utils.second)[0]


    def guess_probability_of_optimality_naive(self, solution: FullSolution) -> float:
        matching_rules = self.rule_manager.get_matching_rules(solution)
        guessed_classes = [rule.predicted_class for rule in matching_rules]
        return np.average(guessed_classes)

    def guess_probability_of_optimality_weighted(self, solution: FullSolution) -> float:
        #only works for binary
        matching_rules = self.rule_manager.get_matching_rules(solution)

        if len(matching_rules) == 0:
            return 6.0

        def get_rule_weight(rule: Rule) -> float:
            return rule.numerosity * rule.get_accuracy()

        action_counts = defaultdict(float)
        for rule in matching_rules:
            action_counts[rule.predicted_class] = action_counts[rule.predicted_class] + get_rule_weight(rule)

        return action_counts[1]/(action_counts[0]+action_counts[1])
    def guess_class(self, solution: FullSolution, method:Literal["naive", "weighted"] = "weighted") -> int:
        if method == "naive":
            return self.guess_class_naive(solution)
        elif method == "weighted":
            return self.guess_class_weighted_on_numerosity_and_accuracy(solution)
        else:
            raise NotImplementedError(f"The method {method} hasn't been implemented yet")

    def guess_probability_of_optimality(self,
                                        solution: FullSolution,
                                        method: Literal["naive", "weighted"] = "weighted") -> float:
        if method == "naive":
            return self.guess_probability_of_optimality_naive(solution)
        elif method == "weighted":
            return self.guess_probability_of_optimality_weighted(solution)
        else:
            raise NotImplementedError(f"The method {method} hasn't been implemented yet")




