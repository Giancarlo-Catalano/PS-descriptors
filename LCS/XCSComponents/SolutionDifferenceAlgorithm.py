import random

import xcs
from xcs import scenarios

import utils
from Core.EvaluatedFS import EvaluatedFS
from Core.PS import PS
from LCS.Conversions import condition_to_ps
from LCS.XCSComponents.SolutionDifferenceModel import SolutionDifferenceModel
from ThirdPaper.SolutionDifferencePSSearch import find_ps_in_solution
from LCS.XCSComponents.SolutionDifferenceScenario import GenericSolutionDifferenceScenario
from LCS.PSEvaluator import GeneralPSEvaluator
from LCS.XCSComponents.CombinatorialRules import CombinatorialCondition


class SolutionDifferenceAlgorithm(xcs.XCSAlgorithm):
    """ This class exists mainly to override the following mechanisms:
    * deciding when covering is required: when most of the solution is 'uncovered'
    * covering using a small NSGAII run [ which also produces more than one covering rule ]"""

    ps_evaluator: GeneralPSEvaluator  # to evaluate the linkage of a rule
    covering_search_budget: int
    covering_population_size: int
    search_for_negative_traits: bool
    xcs_problem: GenericSolutionDifferenceScenario

    verbose: bool
    verbose_search: bool

    def __init__(self,
                 ps_evaluator: GeneralPSEvaluator,
                 xcs_problem: GenericSolutionDifferenceScenario,
                 covering_search_budget: int = 1000,
                 covering_population_size: int = 100,
                 search_for_negative_traits: bool = False,
                 verbose: bool = False,
                 verbose_search: bool = False,
                 ):
        self.ps_evaluator = ps_evaluator
        self.xcs_problem = xcs_problem
        self.covering_search_budget = covering_search_budget
        self.covering_population_size = covering_population_size
        self.search_for_negative_traits = search_for_negative_traits
        self.verbose = verbose
        self.verbose_search = verbose_search

        self.minimum_actions = 1  # otherwise it causes behaviour which I don't understand in XCSAlgorithm.covering_is_required
        super().__init__()


    def get_pss_for_pair(self,
                         winner: EvaluatedFS,
                         loser: EvaluatedFS,
                         only_return_least_dependent:bool = False,
                         only_return_biggest: bool = False) -> list[PS]:

        if self.verbose:
            print(
                f"Covering ({'negative' if self.search_for_negative_traits else 'positive'}) for {winner = }, {loser = }")

        # search for the appropriate patterns using NSGAII (using Pymoo)
        with utils.announce("Mining the PSs...\n", self.verbose_search):
            # debug

            # end of debug
            pss = find_ps_in_solution(to_explain=loser if self.search_for_negative_traits else winner,
                                      unexplained_mask=loser,
                                      population_size=self.covering_population_size,
                                      ps_evaluator=self.ps_evaluator,
                                      ps_budget=self.covering_search_budget,
                                      culling_method="least_dependent",
                                      verbose=self.verbose_search)

            assert (len(pss) > 0)
            return pss

    def cover_with_many(self, match_set: xcs.MatchSet, only_return_one: bool = False) -> list[xcs.ClassifierRule]:
        """ This is a replacement for the .cover function.

        The results must:
        * 1: match the winner
        * 2: NOT match the loser
        *   --> contain at least one part of the difference between them
        """

        # get the PSs in the action set
        winner, loser = match_set.situation
        # self.ps_evaluator.set_solution(winner)
        difference_mask = winner.values != loser.values

        if self.verbose:
            print(f"Covering ({'negative' if self.search_for_negative_traits else 'positive'}) for {winner = }, {loser = }")

        pss = self.get_pss_for_pair(winner, loser, only_return_least_dependent=False, only_return_biggest=True)

        def ps_to_rule(ps: PS) -> xcs.XCSClassifierRule:
            return xcs.XCSClassifierRule(
                CombinatorialCondition.from_ps_values(ps.values),  # NOTE that this is the customised rule
                True,
                self,
                match_set.model.time_stamp)

        if self.verbose_search:
            print('The mined pss are')
            for ps in pss:
                print("\t", ps)

        return list(map(ps_to_rule, pss))

    def new_model(self, scenario):
        # modified because it needs to return an instance of CustomXCSClassifier
        assert isinstance(scenario, scenarios.Scenario)
        return SolutionDifferenceModel(algorithm = self, want_negative_traits = self.search_for_negative_traits)

    def traditional_subsumption(self,
                                action_set):  # this is identical to the original in XCSAlgorithm, but with some more printing
        """Perform action set subsumption."""
        # Select a condition with maximum bit count among those having
        # sufficient experience and sufficiently low error.

        selected_rule = None
        selected_bit_count = None
        for rule in action_set:
            if not (rule.experience > self.subsumption_threshold and
                    rule.error < self.error_threshold):
                continue
            bit_count = rule.condition.count()
            if (selected_rule is None or
                    bit_count > selected_bit_count or
                    (bit_count == selected_bit_count and
                     random.randrange(2))):
                selected_rule = rule
                selected_bit_count = bit_count

        # If no rule was found satisfying the requirements, return
        # early.
        if selected_rule is None:
            return

        # Subsume each rule which the selected rule generalizes. When a
        # rule is subsumed, all instances of the subsumed rule are replaced
        # with instances of the more general one in the population.
        to_remove = []
        for rule in action_set:
            if (selected_rule is not rule and
                    selected_rule.condition(rule.condition)):
                selected_rule.numerosity += rule.numerosity
                action_set.model.discard(rule, rule.numerosity)
                to_remove.append(rule)
                if self.verbose:
                    big_fish = condition_to_ps(selected_rule.condition)
                    small_fish = condition_to_ps(rule.condition)
                    print(f"\t{big_fish}(err = {selected_rule.error}) consumed {small_fish}(err = {rule.error})")
        for rule in to_remove:
            action_set.remove(rule)

    def custom_subsumption(self, action_set):
        eligible_rules = [rule for rule in action_set
                          if rule.experience > self.subsumption_threshold
                          if rule.error < self.error_threshold]
        if len(eligible_rules) == 0:
            return

        # select the rule with the highest bit count
        winning_rule = max(eligible_rules, key=lambda x: x.condition.count())

        def should_be_removed(rule) -> bool:
            return (rule is not winning_rule) and \
                winning_rule.condition(rule.condition) and \
                rule.fitness < winning_rule.fitness

        rules_to_remove = []

        def mark_for_removal(rule) -> None:
            winning_rule.numerosity += rule.numerosity
            action_set.model.discard(rule, rule.numerosity)
            rules_to_remove.append(rule)

        for rule in action_set:
            if should_be_removed(rule):
                mark_for_removal(rule)
                if self.verbose:
                    big_fish = condition_to_ps(winning_rule.condition)
                    small_fish = condition_to_ps(rule.condition)
                    print(
                        f"\t{big_fish}(acc = {winning_rule.accuracy:.2f}) consumed {small_fish}(acc = {rule.accuracy:.2f})")

        for rule in rules_to_remove:
            action_set.remove(rule)

    def _action_set_subsumption(self, action_set):
        """Perform action set subsumption."""
        self.custom_subsumption(action_set)

    def update_attributes_of_rule(self,
                                  rule: xcs.ClassifierRule,
                                  action_set_size: int,
                                  payoff: float):
        # modification of a section in XCSAlgorithm.distribute_payoff
        rule.experience += 1  # appears to be equivalent to the count of matches

        update_rate = max(self.learning_rate, 1 / rule.experience)

        rule.average_reward += (payoff - rule.average_reward) * update_rate

        rule.error += (abs(payoff - rule.average_reward) - rule.error) * update_rate

        rule.action_set_size += (action_set_size - rule.action_set_size) * update_rate

        # custom part

        if not hasattr(rule, "correct_match_count"):
            rule.correct_match_count = 0
        rule.correct_match_count += int(payoff)

        rule.accuracy = rule.correct_match_count / rule.experience

        # end of custom part

    def update_action_set_fitnesses_as_accuracy(self, action_set):
        for rule in action_set:
            rule.fitness = rule.accuracy

    def update_action_set_fitnesses_traditional(self,
                                                action_set: xcs.ActionSet):
        # modification of XCSAlgorithm._update_fitness

        def get_rule_accuracy(rule):
            if rule.error < self.error_threshold:
                return 1
            else:
                return self.accuracy_coefficient * (rule.error / self.error_threshold) ** -self.accuracy_power

        accuracies = list(map(get_rule_accuracy, action_set))
        total_accuracy = sum(accuracy * rule.numerosity for rule, accuracy in zip(action_set, accuracies))

        if total_accuracy == 0:  # I am scared of using 1e-05, because the model already uses that as a default in many places.
            total_accuracy = 1  # to prevent a division by zero, as implemented in the original

        total_accuracy = total_accuracy or 1

        # then update every rule
        for rule, accuracy in zip(action_set, accuracies):
            proposed_new_fitness = (accuracy * rule.numerosity / total_accuracy - rule.fitness)
            rule.fitness += self.learning_rate * proposed_new_fitness

    def update_action_set_fitnesses(self, action_set: xcs.ActionSet):
        self.update_action_set_fitnesses_as_accuracy(action_set)

    def update_match_set_timestamps(self, match_set: xcs.MatchSet):
        # Modification of the initial part of XCSAlgorithm.update(match_set)
        # The original function does too many things, so I'm breaking it up

        # Increment the iteration counter.
        match_set.model.update_time_stamp()

        def update_action_set_timestamps(action_set: xcs.ActionSet):
            average_time_passed = (
                    match_set.model.time_stamp -
                    self._get_average_time_stamp(action_set)
            )
            if average_time_passed <= self.ga_threshold:
                return

            # Update the time step for each rule to indicate that they were
            # updated by the GA.
            self._set_timestamps(action_set)

        update_action_set_timestamps(match_set[True])
        update_action_set_timestamps(match_set[False])

    def apply_payoff_to_match_set(self,
                                  action_set: xcs.ActionSet,
                                  payoff: float):
        # modification of MatchSet.apply_payoff
        action_set_size = sum(rule.numerosity for rule in action_set)

        # Update the average reward, error, and action set size of each
        # rule participating in the action set.
        for rule in action_set:
            self.update_attributes_of_rule(rule, action_set_size=action_set_size, payoff=payoff)

        # Update the fitness of the rules.
        self.update_action_set_fitnesses(action_set)

        # If the parameters so indicate, perform action set subsumption.
        if self.do_action_set_subsumption:
            self._action_set_subsumption(action_set)

        # this is where the .update methods would be

        # GA reproduction moved to its own optional function

        # end of original .update method

    def introduce_rules_via_reproduction(self, action_set: xcs.ActionSet):

        # This is the second part of the XCSAlgorithm.update function
        # it has not been modified, although condition.crossover_with has been overwritten as a uniform crossover

        # Select two parents from the action set, with probability
        # proportionate to their fitness.
        parent1 = self._select_parent(action_set)
        parent2 = self._select_parent(action_set)

        # With the probability specified in the parameters, apply the
        # crossover operator to the parents. Otherwise, just take the
        # parents unchanged.
        if random.random() < self.crossover_probability:
            condition1, condition2 = parent1.condition.crossover_with(
                parent2.condition
            )
        else:
            condition1, condition2 = parent1.condition.copy(), parent2.condition.copy()

        # Apply the mutation operator to each child, randomly flipping
        # their mask bits with a small probability.
        background_solution = action_set.situation[0]
        condition1.mutate(point_mutation_probability=self.mutation_probability, background_solution=background_solution)
        condition2.mutate(point_mutation_probability=self.mutation_probability, background_solution=background_solution)

        # If the newly generated children are already present in the
        # population (or if they should be subsumed due to GA subsumption)
        # then simply increment the numerosities of the existing rules in
        # the population.
        new_children = []
        for condition in condition1, condition2:
            # If the parameters specify that GA subsumption should be
            # performed, look for an accurate parent that can subsume the
            # new child.
            if self.do_ga_subsumption:
                subsumed = False
                for parent in parent1, parent2:
                    should_subsume = (
                            (parent.experience >
                             self.subsumption_threshold) and
                            parent.error < self.error_threshold and
                            parent.condition(condition)
                    )
                    if should_subsume:
                        if parent in action_set.model:
                            parent.numerosity += 1
                            self.prune(action_set.model)
                        else:
                            # Sometimes the parent is removed from a
                            # previous subsumption
                            parent.numerosity = 1
                            action_set.model.add(parent)
                        subsumed = True
                        break
                if subsumed:
                    continue

            # Provided the child has not already been subsumed and it is
            # present in the population, just increment its numerosity.
            # Otherwise, if the child has neither been subsumed nor does it
            # already exist, remember it so we can add it to the classifier
            # set in just a moment.
            child = xcs.XCSClassifierRule(
                condition,
                action_set.action,
                self,
                action_set.model.time_stamp
            )
            if child in action_set.model:
                action_set.model.add(child)
            else:
                new_children.append(child)

        # If there were any children which weren't subsumed and weren't
        # already present in the classifier set, add them.
        if new_children:
            average_reward = .5 * (
                    parent1.average_reward +
                    parent2.average_reward
            )

            error = .5 * (parent1.error + parent2.error)

            # .1 * (average fitness of parents)
            fitness = .05 * (
                    parent1.fitness +
                    parent2.fitness
            )

            for child in new_children:
                if self.verbose:
                    print(f"Via GA, adding the child {child.condition}")

                child.average_reward = average_reward
                child.error = error
                child.fitness = fitness
                action_set.model.add(child)

    def covering_is_required(self, match_set):
        # Modification of XCSAlgorithm.covering_is_required

        correct_set = match_set[not self.search_for_negative_traits]
        amount_in_correct_action_set = len(correct_set._rules)  # I am really getting annoyed at this library...

        if self.minimum_actions is None:
            return amount_in_correct_action_set < 1
        else:
            return amount_in_correct_action_set < self.minimum_actions

    def prune(self, model: SolutionDifferenceModel) -> list[xcs.ClassifierRule]:
        """ custom reimplementation in order to simplify the pruning process"""
        """ deletes a single rule, which consists in calling model.discard(rule) and returning it into a singleton list"""

        """ the assumption is that the rules' fitnesses are their accuracy (correct / matches)"""

        total_numerosity = sum(rule.numerosity for rule in model)
        if total_numerosity <= self.max_population_size:
            return []

        rules_available_for_removal = [rule for rule in model
                                       if rule.experience > self.deletion_threshold
                                       if rule.fitness < 1.0]
        if len(rules_available_for_removal) == 0:
            if self.verbose:
                print(
                    f"While pruning is necessary (tot.numerosity = {total_numerosity}, #rules = {len(list(model))}), no rules are eligible for removal")
            return []

        to_remove = min(rules_available_for_removal, key=lambda x: x.fitness)
        if self.verbose:
            print(f"In pruning, the condition {to_remove.condition} will be discarded "
                  f"(numerosity = {to_remove.numerosity}), "
                  f"its accuracy was {to_remove.fitness}")
        model.discard(to_remove)
        return [to_remove]

    def toggle_verbose_search(self):
        self.verbose_search = not self.verbose_search
