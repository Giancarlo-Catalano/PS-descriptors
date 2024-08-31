import random

import numpy as np
import xcs
from xcs import scenarios
from xcs.scenarios import Scenario

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from LCS.Conversions import get_pss_from_action_set, get_action_set, \
    ps_to_condition, get_conditions_in_match_set, condition_to_ps
from LCS.CustomXCSClassifierSet import CustomXCSClassiferSet
from LCS.SolutionDifference.SolutionDifferenceModel import SolutionDifferenceModel
from LCS.SolutionDifference.SolutionDifferencePSSearch import local_restricted_tm_ps_search
from LCS.SolutionDifference.SolutionDifferenceScenario import SolutionDifferenceScenario
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator, local_tm_ps_search


class SolutionDifferenceAlgorithm(xcs.XCSAlgorithm):
    """ This class exists mainly to override the following mechanisms:
    * deciding when covering is required: when most of the solution is 'uncovered'
    * covering using a small NSGAII run [ which also produces more than one covering rule ]"""

    ps_evaluator: TMEvaluator  # to evaluate the linkage of a rule
    covering_search_budget: int
    xcs_problem: SolutionDifferenceScenario

    verbose: bool

    def __init__(self,
                 ps_evaluator: TMEvaluator,
                 xcs_problem: SolutionDifferenceScenario,
                 covering_search_budget: int = 1000,
                 verbose: bool = False,
                 ):
        self.ps_evaluator = ps_evaluator
        self.xcs_problem = xcs_problem
        self.covering_search_budget = covering_search_budget
        self.verbose = verbose
        super().__init__()

    def cover_with_many(self, match_set: xcs.MatchSet) -> list[xcs.ClassifierRule]:
        """ This is a replacement for the .cover function.

        The results must:
        * 1: match the winner
        * 2: NOT match the loser
        *   --> contain at least one part of the difference between them


        """

        # get the PSs in the action set
        winner, loser = match_set.situation
        self.ps_evaluator.set_solution(winner)
        difference_mask = winner.values != loser.values

        # search for the appropriate patterns using NSGAII (using Pymoo)
        with utils.announce("Mining the PSs...", True):
            pss = local_restricted_tm_ps_search(to_explain=winner,
                                                pss_to_avoid=[],
                                                must_include_mask=difference_mask,
                                                population_size=50,  # TODO parametrize this
                                                ps_evaluator=self.ps_evaluator,
                                                ps_budget=self.covering_search_budget,
                                                verbose=False)

        assert (len(pss) > 0)

        def ps_to_rule(ps: PS) -> xcs.XCSClassifierRule:
            return xcs.XCSClassifierRule(
                ps_to_condition(ps),
                True,
                self,
                match_set.model.time_stamp)

        return list(map(ps_to_rule, pss))

    def new_model(self, scenario):
        # modified because it needs to return an instance of CustomXCSClassifier
        assert isinstance(scenario, scenarios.Scenario)
        return SolutionDifferenceModel(self, scenario.get_possible_actions())

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
                rule.error > winning_rule.error

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
                    print(f"\t{big_fish}(err = {winning_rule.error}) consumed {small_fish}(err = {rule.error})")

        for rule in rules_to_remove:
            action_set.remove(rule)

    def _action_set_subsumption(self, action_set):
        """Perform action set subsumption."""
        self.traditional_subsumption(action_set)
