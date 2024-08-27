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
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator, local_tm_ps_search


def get_solution_coverage(match_set: xcs.MatchSet, action) -> float:
    """ Checks how much of the situation is covered by the action set, as a percentage """

    if action is not None:
        action_set = get_action_set(match_set, action)
        conditions = list(action_set._rules)  # these are conditions, for some reason
    else:
        rules = get_conditions_in_match_set(match_set)
        conditions = [rule.condition for rule in rules]
    masks = np.array([np.array(condition.mask) for condition in conditions])
    if len(masks) == 0:
        return 0
    covered_vars = np.sum(masks, axis=0, dtype=bool)
    return np.average(covered_vars)


class CustomXCSAlgorithm(xcs.XCSAlgorithm):
    """ This class exists mainly to override the following mechanisms:
    * deciding when covering is requiered: when most of the solution is 'uncovered'
    * covering using a small NSGAII run [ which also produces more than one covering rule ]"""

    ps_evaluator: TMEvaluator  # to evaluate the linkage of a rule
    coverage_covering_threshold: float  # how much a scenario needs to be covered by rules ([0, 1])
    covering_search_budget: int
    xcs_problem: Scenario

    verbose: bool

    def __init__(self,
                 ps_evaluator: TMEvaluator,
                 xcs_problem: Scenario,
                 coverage_covering_threshold: float = 0.5,
                 covering_search_budget: int = 1000,
                 verbose: bool = False,
                 ):
        self.ps_evaluator = ps_evaluator
        self.coverage_covering_threshold = coverage_covering_threshold
        self.xcs_problem = xcs_problem
        self.covering_search_budget = covering_search_budget
        self.verbose = verbose
        super().__init__()

    def covering_is_required(self, match_set: xcs.MatchSet) -> bool:
        action = self.xcs_problem.get_current_outcome()
        coverage = get_solution_coverage(match_set, action = None)  # action = None means that we don't care
        should_cover = coverage < self.coverage_covering_threshold

        if self.verbose:
            print(f"Coverage for {FullSolution(match_set.situation)} (action = {action}) is {int(coverage * 100)}%")
        return should_cover

    def get_appropriate_action_for_ps(self, ps: PS):
        """ Unlike a normal LCS where the action is based on the current datapoint, we find it ourselves"""
        """ The current implementation checks the average fitness of the ps"""
        return self.ps_evaluator.is_ps_beneficial(ps)

    def return_most_appropriate_pss(self,
                                    pss: list[PS],
                                    suggested_action,
                                    consistency_threshold: float = 0.05,
                                    restrict_consistency: bool = True,
                                    restrict_action: bool = True) -> (list[PS], bool, float):

        deltas = [self.ps_evaluator.delta_fitness_metric.get_mean_fitness_delta(ps) for ps in pss]
        actions = [delta > 0 for delta in deltas]
        p_values = [self.ps_evaluator.fitness_p_value_metric.test_effect(ps, supposed_beneficial=suggested_action)
                    for ps, suggested_action in zip(pss, actions)]

        if self.verbose:
            optimisation_problem: BenchmarkProblem = self.xcs_problem.original_problem
            print(
                f"Covering for \n",
                utils.indent(f"{optimisation_problem.repr_fs(self.xcs_problem.current_solution)}, "
                             f"action = {int(suggested_action)}, yielded:"))
            for ps, delta, action, p_value in zip(pss, deltas, actions, p_values):
                if ps.is_empty():
                    continue
                a, d = ps.metric_scores
                print(utils.indent(
                    f"{optimisation_problem.repr_ps(ps)} -> {int(action)},"
                    f" internal = {-a:.3f}, external = {-d:.3f},"
                    f" delta = {delta:.3f},"
                    f" {p_value = :.3f}"
                    f" {'(DISAGREES)' if action != suggested_action else ''}"
                    f" {'(INCONSISTENT)' if p_value > consistency_threshold else ''})"))

        items = list(zip(pss, actions, p_values))
        items = [item for item in items if not item[0].is_empty()] # remove the empty PS
        if len(items) < 1:
            raise Exception("There should be at least one ps found!")

        if restrict_consistency:
            # remove the entries with bad consistency
            good_consistency = [(ps, action, p_value)
                                for (ps, action, p_value) in items if p_value < consistency_threshold]

            if len(good_consistency) > 1:
                items = good_consistency  # but never allow items to be empty!
            elif self.verbose:
                    print("Ignoring consistency constraint to prevent empty covering")

        if restrict_action:
            # remove the entries with the wrong action
            correct_action = [(ps, action, p_value)
                              for (ps, action, p_value) in items if action == suggested_action]

            if len(correct_action) > 1:
                items = correct_action  # but never allow items to be empty!
            elif self.verbose:
                print("Ignoring action constraint to prevent empty covering")

        return items

    def cover_with_many(self, match_set: xcs.MatchSet) -> list[xcs.ClassifierRule]:
        """ This is a replacement for the .cover function. The main difference is that this returns many rules"""

        # get the PSs in the action set
        suggested_action = self.xcs_problem.get_current_outcome()
        action_set = get_action_set(match_set, suggested_action)
        already_found_pss = get_pss_from_action_set(action_set)

        # set the linkage evaluator to produce results relating to the solution
        solution = FullSolution(match_set.situation)
        self.ps_evaluator.set_solution(solution)

        # search for the appropriate patterns using NSGAII (using Pymoo)
        with utils.announce("Mining the PSs...", True):
            pss = local_tm_ps_search(to_explain=solution,
                                     to_avoid=[], # already_found_pss, # uncomment to avoid already found pss
                                     population_size=50, # TODO parametrize this
                                     ps_evaluator=self.ps_evaluator,
                                     ps_budget=self.covering_search_budget,
                                     verbose=False)

        # find the most appropriate actions for each new rule
        eligible_items = self.return_most_appropriate_pss(pss, suggested_action, restrict_action=False)
        assert(len(eligible_items) > 0)

        def ps_to_rule(ps: PS, action) -> xcs.XCSClassifierRule:
            return xcs.XCSClassifierRule(
                ps_to_condition(ps),
                action,
                self,
                match_set.model.time_stamp)

        return [ps_to_rule(ps, action)
                for ps, action, p_value in eligible_items]

    def new_model(self, scenario):
        # modified because it needs to return an instance of CustomXCSClassifier
        assert isinstance(scenario, scenarios.Scenario)
        return CustomXCSClassiferSet(self, scenario.get_possible_actions())


    def _action_set_subsumption(self, action_set):
        """Perform action set subsumption."""
        # this is identical to the original in XCSAlgorithm, but with some more printing
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
                    print(f"\t{big_fish} consumed {small_fish}")
        for rule in to_remove:
            action_set.remove(rule)
