import numpy as np

from Core.FullSolution import FullSolution
from Core.PS import STAR
from OwnLCS.RuleEvolver import RuleEvolver
from OwnLCS.Rule import Rule, OutputClass


class RuleManager:
    rule_population: list[Rule]
    rule_evolver: RuleEvolver

    training_iteration: int

    evolve_new_individuals_interval: int
    truncation_selection_interval: int
    subsumption_interval: int

    def __init__(self,
                 rule_evolver: RuleEvolver,
                 rule_population: list[Rule] = None,
                 evolve_new_individuals_interval: int = 1,
                 truncation_selection_interval: int = 20,
                 subsumption_interval: int = 20):
        self.rule_evolver = rule_evolver

        self.rule_population = [] if rule_population is None else rule_population

        self.training_iteration = 0

        self.evolve_new_individuals_interval = evolve_new_individuals_interval
        self.truncation_selection_interval = truncation_selection_interval
        self.subsumption_interval = subsumption_interval

    def get_matching_rules(self, full_solution: FullSolution) -> list[Rule]:
        return [rule for rule in self.rule_population if rule.matches(full_solution)]

    def train_on_instance(self,
                          full_solution: FullSolution,
                          output_class: OutputClass):
        self.training_iteration += 1

        match_set = self.get_matching_rules(full_solution)
        correct_set = [rule for rule in match_set if rule.predicted_class == output_class]

        if len(correct_set) == 0:  # covering
            new_rules = self.apply_covering(full_solution, output_class)
            self.rule_population.extend(new_rules)
            match_set.extend(new_rules)
            correct_set = new_rules

        # update accuracy
        for rule in match_set:
            rule.amount_of_matches += 1

        for rule in correct_set:
            rule.amount_of_correct += 1

        # adds 2 new individuals
        if (self.training_iteration) % self.evolve_new_individuals_interval == 0:
            self.rule_population = self.rule_evolver.online_evolution_step(self.rule_population, match_set, correct_set)

        # applies subsumption, ie removes redundant rules
        if (self.training_iteration) % self.subsumption_interval == 0:
            self.rule_population = self.apply_subsumption(self.rule_population)

        # does truncation selection
        if (self.training_iteration) % self.truncation_selection_interval == 0:
            self.rule_population = self.rule_evolver.restrict_population_size(self.rule_population)



    def get_amount_of_rules_for_covering(self) -> int:
        return self.rule_evolver.search_space.amount_of_parameters

    def apply_covering(self, full_solution: FullSolution, output_class: OutputClass) -> list[Rule]:
        amount_of_new_rules = self.get_amount_of_rules_for_covering()
        new_rules = [self.rule_evolver.get_random_individual_using_covering(full_solution, output_class)
                     for _ in range(amount_of_new_rules)]
        return new_rules

    def apply_training_batch(self, instances: list[(FullSolution, OutputClass)]):
        for solution, actual_class in instances:
            self.train_on_instance(solution, actual_class)

    def apply_subsumption(self, population: list[Rule]):
        # first we sort the rules by accuracy
        sorted_rules = sorted(population, key=lambda x: x.get_accuracy(), reverse=False)

        def eats(eater: Rule, eaten: Rule) -> bool:
            def ps_match():
                return np.all((eater.ps.values == STAR) | (eater.ps.values == eaten.ps.values))

            def more_accurate():
                return eater.get_accuracy() >= eaten.get_accuracy()

            def same_action():
                return eater.predicted_class == eaten.predicted_class

            return same_action() and more_accurate() and ps_match()  # order of slowness

        new_rules = []
        for eaten_index, eaten_rule in enumerate(sorted_rules):
            was_eaten = False
            for eater_rule in sorted_rules[eaten_index + 1:]:
                if eats(eater_rule, eaten_rule):
                    eater_rule.numerosity += eaten_rule.numerosity
                    was_eaten = True
                    # print(f"{eater_rule} ate {eaten_rule}")
                    break
            if not was_eaten:
                new_rules.append(eaten_rule)

        return new_rules

    def apply_covering_with_evolution(self, solution: FullSolution, output_class: OutputClass) -> list[Rule]:

        covering_population = []
        population_size = 30
        while len(covering_population) < population_size:
            covering_population.extend(self.apply_covering(solution, output_class))

        def with_mutated_offspring(population: list[Rule]) -> list[Rule]:
            mutated = [self.rule_evolver.mutated(rule) for rule in population]
            return list(set(population + mutated))

        generations = 50
        for generation in range(generations):
            covering_population = with_mutated_offspring(covering_population)
            covering_population = self.rule_evolver.restrict_population_size(covering_population)

        return self.rule_evolver.restrict_population_size(covering_population, self.get_amount_of_rules_for_covering())
