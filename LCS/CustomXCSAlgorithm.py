import xcs
from xcs.bitstrings import BitCondition

import utils
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LCS.Conversions import get_solution_coverage, situation_to_fs, get_pss_from_action_set, get_action_set, \
    ps_to_condition
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search
from utils import sort_by_combination_of


# model.run()
#   self.match(situation)
#       if self._algorithm.covering_is_required(match_set): <- override
#       self._algorithm.cover(match_set)  <- override
class CustomXCSAlgorithm(xcs.XCSAlgorithm):

    ps_evaluator: FastPSEvaluator
    coverage_covering_threshold: float
    xcs_problem: XCSProblemTournamenter

    def __init__(self,
                 ps_evaluator: FastPSEvaluator,
                 xcs_problem: XCSProblemTournamenter,
                 coverage_covering_threshold: float = 0.5,
                 ):
        self.ps_evaluator = ps_evaluator
        self.coverage_covering_threshold = coverage_covering_threshold
        self.xcs_problem = xcs_problem
        super().__init__()


    def covering_is_required(self, match_set: xcs.MatchSet) -> bool:
        return get_solution_coverage(match_set, self.xcs_problem.is_current_better) < self.coverage_covering_threshold


    def cover(self, match_set: xcs.MatchSet) -> xcs.ClassifierRule:
        #return super().cover(match_set)
        action = self.xcs_problem.is_current_better
        action_set = get_action_set(match_set, action)
        already_found_pss = get_pss_from_action_set(action_set)
        solution = FullSolution(match_set.situation)
        if action:
            self.ps_evaluator.set_to_maximise_fitness()
        else:
            self.ps_evaluator.set_to_minimise_fitness()

        fitness = self.xcs_problem.original_problem.fitness_function(solution) # TODO recycle it from before
        self.ps_evaluator.atomicity_metric.set_solution(EvaluatedFS(solution, fitness))

        pss = local_ps_search(to_explain = solution,
                              to_avoid=already_found_pss,
                              population_size=50,
                              ps_evaluator=self.ps_evaluator,
                              ps_budget = 500,
                              verbose=False)
        winning_ps = min(pss, key=lambda x: x.metric_scores[1]) # there's some messing with the signs
        # pss = sort_by_combination_of(pss, key_functions=[lambda x: x.metric_scores[0],
        #                                                  lambda x: x.metric_scores[1],
        #                                                  lambda x: x.metric_scores[2]])
        #winning_ps = pss[0]

        print(f"Covering for {self.xcs_problem.current_solution}, action = {int(action)}, yielded {winning_ps}")

        return xcs.XCSClassifierRule(
            ps_to_condition(winning_ps),
            action,
            self,
            match_set.model.time_stamp)

