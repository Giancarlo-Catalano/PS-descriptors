import xcs
from xcs.bitstrings import BitCondition

from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from LCS.Conversions import get_solution_coverage, situation_to_fs, get_pss_from_action_set
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search


# model.run()
#   self.match(situation)
#       if self._algorithm.covering_is_required(match_set): <- override
#       self._algorithm.cover(match_set)  <- override
class CustomXCSAlgorithm(xcs.XCSAlgorithm):

    ps_evaluator: FastPSEvaluator
    coverage_covering_threshold: float

    def __init__(self,
                 ps_evaluator: FastPSEvaluator,
                 coverage_covering_threshold: float = 0.5):
        self.ps_evaluator = ps_evaluator
        self.coverage_covering_threshold = coverage_covering_threshold
        super().__init__()


    def covering_is_required(self, match_set: xcs.MatchSet) -> bool:
        return get_solution_coverage(match_set) < self.coverage_covering_threshold


    def cover(self, match_set: xcs.MatchSet) -> xcs.ClassifierRule:
        print("Attemping to cover for a match set")
        #return super().cover(match_set)
        action = True # TODO
        action_set = match_set[match_set.situation.action]
        already_found_pss = get_pss_from_action_set(action_set)
        solution = FullSolution(match_set.situation)
        if action:
            self.ps_evaluator.set_to_maximise_fitness()
        else:
            self.ps_evaluator.set_to_minimise_fitness()

        pss = local_ps_search(to_explain = solution,
                              to_avoid=already_found_pss,
                              population_size=50,
                              ps_evaluator=self.ps_evaluator,
                              ps_budget = 300,
                              verbose=False)
        winning_ps = pss[0]

        bits = winning_ps.values.copy()
        mask = winning_ps.values != STAR
        bits[~mask] = 0

        return xcs.XCSClassifierRule(
            BitCondition(bits, mask),
            action,
            self,
            match_set.model.time_stamp)

