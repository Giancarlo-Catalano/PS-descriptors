import numpy as np
import xcs
from xcs import scenarios
from xcs.scenarios import Scenario

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from LCS.Conversions import get_pss_from_action_set, get_action_set, \
    ps_to_condition
from LCS.CustomXCSClassifierSet import CustomXCSClassiferSet
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator, local_tm_ps_search


def get_solution_coverage(match_set: xcs.MatchSet, action) -> float:
    """ Checks how much of the situation is covered by the action set, as a percentage """
    action_set = get_action_set(match_set, action)
    rules = list(action_set._rules)  # these are conditions, for some reason
    masks = np.array([np.array(condition.mask) for condition in rules])
    if len(masks) == 0:
        return 0
    covered_vars = np.sum(masks, axis=0, dtype=bool)
    return np.average(covered_vars)

class CustomXCSAlgorithm(xcs.XCSAlgorithm):
    """ This class exists mainly to override the following mechanisms:
    * deciding when covering is requiered: when most of the solution is 'uncovered'
    * covering using a small NSGAII run [ which also produces more than one covering rule ]"""

    ps_evaluator: TMEvaluator   # to evaluate the linkage of a rule
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
        coverage = get_solution_coverage(match_set, self.xcs_problem.get_current_outcome())
        should_cover = coverage < self.coverage_covering_threshold

        if self.verbose:
            print(f"Coverage for {FullSolution(match_set.situation)} is {int(coverage * 100)}%")
        return should_cover

    def get_appropriate_action_for_ps(self, ps: PS):
        """ Unlike a normal LCS where the action is based on the current datapoint, we find it ourselves"""
        """ The current implementation checks the average fitness of the ps"""
        return self.ps_evaluator.is_ps_beneficial(ps)

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
        pss = local_tm_ps_search(to_explain=solution,
                                 to_avoid=already_found_pss,
                                 population_size=30,
                                 ps_evaluator=self.ps_evaluator,
                                 ps_budget=1000,
                                 verbose=False)

        # find the most appropriate actions for each new rule
        actions = [self.get_appropriate_action_for_ps(ps) for ps in pss]

        if self.verbose:
            optimisation_problem: BenchmarkProblem = self.xcs_problem.original_problem
            print(
                f"Covering for {optimisation_problem.repr_fs(self.xcs_problem.current_solution)}, action = {int(suggested_action)}, yielded:")
            for ps, ps_action in zip(pss, actions):
                a, d = ps.metric_scores
                delta = self.ps_evaluator.mean_fitness_metric.get_single_score(ps)
                print(
                    f"\t{optimisation_problem.repr_ps(ps)} -> {int(ps_action)},"
                    f" internal = {a:.3f}, external = {d:.3f},"
                    f" ({delta = :.3f} "
                    f" {'(DISAGREES)' if ps_action != suggested_action else ''})"
                )

        def ps_to_rule(ps: PS, action) -> xcs.XCSClassifierRule:
            return xcs.XCSClassifierRule(
                ps_to_condition(ps),
                action,
                self,
                match_set.model.time_stamp)

        return [ps_to_rule(ps, action)
                for ps, action in zip(pss, actions)
                # if action == suggested_action  # if you uncomment this line, you only allow rules that match the action
                ]

    def new_model(self, scenario):
        # modified because it needs to return an instance of CustomXCSClassifier
        assert isinstance(scenario, scenarios.Scenario)
        return CustomXCSClassiferSet(self, scenario.get_possible_actions())
