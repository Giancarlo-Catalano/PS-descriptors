import numpy as np
import xcs
from xcs import scenarios
from xcs.scenarios import Scenario

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from LCS.Conversions import get_pss_from_action_set, get_action_set, \
    ps_to_condition, get_conditions_in_match_set
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
    xcs_problem: Scenario

    verbose: bool

    def __init__(self,
                 ps_evaluator: TMEvaluator,
                 xcs_problem: Scenario,
                 coverage_covering_threshold: float = 0.5,
                 verbose: bool = False,
                 ):
        self.ps_evaluator = ps_evaluator
        self.coverage_covering_threshold = coverage_covering_threshold
        self.xcs_problem = xcs_problem
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
                                    consistency_threshold: float = 0.05) -> (list[PS], bool, float):

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
                a, d = ps.metric_scores
                print(utils.indent(
                    f"{optimisation_problem.repr_ps(ps)} -> {int(action)},"
                    f" internal = {-a:.3f}, external = {-d:.3f},"
                    f" delta = {delta:.3f},"
                    f" {p_value = :.3f}"
                    f" {'(DISAGREES)' if action != suggested_action else ''}"
                    f" {'(INCONSISTENT)' if p_value > consistency_threshold else ''})"))

        items = list(zip(pss, actions, p_values))
        if len(items) < 1:
            raise Exception("There should be at least one ps found!")

        # remove the entries with bad consistency
        good_consistency = [(ps, action, p_value)
                            for (ps, action, p_value) in items if p_value < consistency_threshold]

        if len(good_consistency) > 1:
            items = good_consistency  # but never allow items to be empty!
        elif self.verbose:
                print("Ignoring consistency constraint to prevent empty covering")

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
                                     to_avoid=already_found_pss,
                                     population_size=30,
                                     ps_evaluator=self.ps_evaluator,
                                     ps_budget=500,
                                     verbose=True)

        # find the most appropriate actions for each new rule
        eligible_items = self.return_most_appropriate_pss(pss, suggested_action)
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
