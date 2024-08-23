import heapq
from typing import Any

import numpy as np
import xcs
from xcs.bitstrings import BitString, BitCondition
from xcs.scenarios import Scenario, ScenarioObserver

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search


def condition_to_ps(bitcondition: BitCondition) -> PS:
    bits = bitcondition.bits
    mask = bitcondition.mask

    ps_values = np.array(bits)
    where_unset = np.logical_not(np.array(mask, dtype=bool))
    ps_values[where_unset] = STAR
    return PS(ps_values)

def rule_to_ps(rule: xcs.ClassifierRule) -> PS:
    return condition_to_ps(rule.condition)


def ps_to_condition(ps: PS) -> BitCondition:
    bits = ps.values.copy()
    mask = ps.values != STAR
    bits[~mask] = 0

    return BitCondition(bits, mask)
def ps_to_rule(algorithm,
                  ps :PS,
                  action) -> xcs.ClassifierRule:
    return xcs.XCSClassifierRule(
        ps_to_condition(ps),
        action,
        algorithm,
        0)

def situation_to_fs(situation) -> FullSolution:
    return FullSolution(situation)



def get_pss_from_action_set(action_set: xcs.ActionSet) -> list[PS]:
    rules = action_set._rules
    return list(map(condition_to_ps, rules))

def get_rules_in_model(model: xcs.ClassifierSet) -> list[(PS, Any)]:
    result = model._population

    pss = map(condition_to_ps, result)  # result is a dictionary, where the keys are bitconditions. We convert each to a ps
    actions = [list(assigned_actions) for assigned_actions in result.values()]

    return list(zip(pss, actions))





def get_action_set(match_set: xcs.MatchSet, action) -> xcs.ActionSet:
    def make_empty_action_set():
        return xcs.ActionSet(model = match_set.model,
                             situation=match_set.situation,
                             action=action,
                             rules = dict())
    return match_set._action_sets.get(action, make_empty_action_set())