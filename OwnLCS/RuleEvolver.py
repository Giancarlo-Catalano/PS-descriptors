import copy
import random
from typing import TypeAlias

from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from OwnLCS.Rule import Rule
from OwnLCS.RuleEvaluator import RuleEvaluator
from utils import sort_by_combination_of

OutputClass: TypeAlias = int
class RuleEvolver:
    population_size: int
    search_space: SearchSpace
    rule_evaluator: RuleEvaluator

    mutation_rate: float
    tournament_size: int


    def __init__(self,
                 population_size: int,
                 search_space: SearchSpace,
                 rule_evaluator: RuleEvaluator,
                 mutation_rate: float = None,
                 tournament_size = 3):
        self.population_size = population_size
        self.search_space = search_space
        self.rule_evaluator = rule_evaluator

        if mutation_rate is None:
            mutation_rate = 1/self.search_space.amount_of_parameters
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size


    def with_cached_metrics(self, rule: Rule) -> Rule:
        rule.cached_linkage = self.rule_evaluator.linkage_metric.get_single_score(rule.ps)
        rule.cached_simplicity = self.rule_evaluator.simplicity_metric.get_single_score(rule.ps)
        return rule

    def get_random_individual_using_covering(self,
                                             full_solution: FullSolution,
                                             actual_class: OutputClass):
        resulting_rule = PS.empty(self.search_space)
        probability_of_cell_assignment = 0.7
        while random.random() < probability_of_cell_assignment:
            active_variable_index = random.randrange(self.search_space.amount_of_parameters)
            variable_value = full_solution.values[active_variable_index]
            resulting_rule = resulting_rule.with_fixed_value(active_variable_index, variable_value)

        return self.with_cached_metrics(Rule(resulting_rule, actual_class, amount_of_matches=1, amount_of_correct=1))


    def sort_population(self, population: list[Rule]) -> list[Rule]:
        metrics = [self.rule_evaluator.get_metrics(rule) for rule in population]
        items_with_metrics = list(zip(population, metrics))
        sorted_items_with_metrics = sort_by_combination_of(items_with_metrics,
                                                           key_functions=[
                                                               lambda x: x[1][0],
                                                               lambda x: x[1][1],
                                                               lambda x: x[1][2]],
                                                           reverse=True)
        return [rule for rule, metrics in sorted_items_with_metrics]
    def restrict_population_size(self, population: list[Rule], to_size = None):
        metrics = [self.rule_evaluator.get_metrics(rule) for rule in population]
        items_with_metrics = list(zip(population, metrics))
        sorted_items_with_metrics = sort_by_combination_of(items_with_metrics,
                               key_functions = [
                                   lambda x: x[1][0],
                                   lambda x: x[1][1],
                                   lambda x: x[1][2]],
                               reverse=True)

        final_size = self.population_size if to_size is None else to_size
        return [rule for rule, metrics in sorted_items_with_metrics[:final_size]]

    def select(self, population: list[Rule]):
        # assumes that the population is sorted from best to worst
        popsize = len(population)
        best_index = min(random.randrange(popsize) for _ in range(self.tournament_size))
        return population[best_index]

    def mutated(self, rule: Rule) -> Rule:
        offspring_ps = rule.ps.copy()
        # force at least one mutation
        mutation_position = random.randrange(self.search_space.amount_of_parameters)
        offspring_ps[mutation_position] = self.search_space.random_digit(mutation_position)
        for index, value in enumerate(offspring_ps.values):
            if random.random() < self.mutation_rate:
                if value == STAR:
                    new_value = self.search_space.random_digit(index)
                else:
                    new_value = STAR
                offspring_ps[index] = new_value
        return self.with_cached_metrics(Rule(offspring_ps, rule.predicted_class))


    def introduce_mutated_individuals(self, population: list[Rule]) -> list[Rule]:
        qty_new_individuals = self.population_size //  12
        def make_new_individual() -> Rule:
            return self.mutated(self.select(population))

        return population + [make_new_individual() for _ in range(qty_new_individuals)]

    def online_evolution_step(self,
                              rule_population: list[Rule],
                              match_set: list[Rule],
                              correct_set: list[Rule]) -> list[Rule]:
        correct_set = self.sort_population(correct_set)

        # select parents
        mother = self.select(correct_set)
        father = self.select(correct_set)
        daughter, son = self.crossover(mother, father)
        daughter = self.mutated(daughter)
        son = self.mutated(son)

        potential_new_population = rule_population + [daughter, son]
        return potential_new_population

    def crossover(self, mother: Rule, father: Rule) -> (Rule, Rule):
        # uniform crossover
        daughter_values = mother.ps.values.copy()
        son_values = father.ps.values.copy()

        for index in range(len(mother.ps.values)):
            if random.random() < 0.5:
                daughter_values[index], son_values[index] = son_values[index], daughter_values[index]

        daughter = self.with_cached_metrics(Rule(PS(daughter_values), mother.predicted_class))
        son = self.with_cached_metrics(Rule(PS(son_values), father.predicted_class))
        return daughter, son


