import xcs
from xcs import scenarios
from xcs.scenarios import Scenario

from Core.FullSolution import FullSolution
from Core.PS import PS
from LCS.Conversions import get_solution_coverage, get_pss_from_action_set, get_action_set, \
    ps_to_condition
from LCS.CustomXCSClassifierSet import CustomXCSClassiferSet
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator, local_tm_ps_search


# model.run()
#   self.match(situation)
#       if self._algorithm.covering_is_required(match_set): <- override
#       self._algorithm.cover(match_set)  <- override
class CustomXCSAlgorithm(xcs.XCSAlgorithm):

    ps_evaluator: TMEvaluator
    coverage_covering_threshold: float
    xcs_problem: Scenario

    def __init__(self,
                 ps_evaluator: TMEvaluator,
                 xcs_problem: Scenario,
                 coverage_covering_threshold: float = 0.5,
                 ):
        self.ps_evaluator = ps_evaluator
        self.coverage_covering_threshold = coverage_covering_threshold
        self.xcs_problem = xcs_problem
        super().__init__()


    def covering_is_required(self, match_set: xcs.MatchSet) -> bool:
        coverage = get_solution_coverage(match_set, self.xcs_problem.get_current_outcome())
        should_cover = coverage < self.coverage_covering_threshold

        situation = FullSolution(match_set.situation)
        print(f"Coverage for {situation} is {int(coverage*100)}%")
        return should_cover


    def cover_with_many(self, match_set: xcs.MatchSet) -> list[xcs.ClassifierRule]:
        #return super().cover(match_set)
        action = self.xcs_problem.get_current_outcome()
        action_set = get_action_set(match_set, action)
        already_found_pss = get_pss_from_action_set(action_set)
        solution = FullSolution(match_set.situation)
        self.ps_evaluator.set_solution(solution)

        pss = local_tm_ps_search(to_explain = solution,
                              to_avoid=already_found_pss,
                              population_size=50,
                              ps_evaluator=self.ps_evaluator,
                              ps_budget = 1000,
                              verbose=False)

        print(f"Covering for {self.xcs_problem.current_solution}, action = {int(action)}, yielded {pss}")

        def ps_to_rule(ps: PS) -> xcs.XCSClassifierRule:
            return xcs.XCSClassifierRule(
            ps_to_condition(ps),
            action,
            self,
            match_set.model.time_stamp)

        return list(map(ps_to_rule, pss))


    def new_model(self, scenario):
        # modified because it needs to return an instance of CustomXCSClassifier
        assert isinstance(scenario, scenarios.Scenario)
        return CustomXCSClassiferSet(self, scenario.get_possible_actions())





