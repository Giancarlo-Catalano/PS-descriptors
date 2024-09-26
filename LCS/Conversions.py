import numpy as np
import xcs
from xcs.bitstrings import BitCondition

from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from LCS.XCSComponents.CombinatorialRules import CombinatorialCondition

""" This file is where I put the functions to convert between PSs and classifier rules"""


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
               ps: PS,
               action) -> xcs.ClassifierRule:
    return xcs.XCSClassifierRule(
        ps_to_condition(ps),
        action,
        algorithm,
        0)


def situation_to_fs(situation) -> FullSolution:
    return FullSolution(situation)


def get_pss_from_action_set(action_set: xcs.ActionSet) -> list[PS]:
    rules = action_set._rules  # yes I access private members, what about it
    return list(map(condition_to_ps, rules))


def get_rules_in_model(model: xcs.ClassifierSet) -> list[xcs.XCSClassifierRule]:
    return list(model)


def get_action_set(match_set: xcs.MatchSet, action) -> xcs.ActionSet:
    # this function exists because empty action sets are annoying to handle
    """ Returns the action set from the provided match set, and returns an empty action set if appropriate"""

    def make_empty_action_set():
        return xcs.ActionSet(model=match_set.model,
                             situation=match_set.situation,
                             action=action,
                             rules=dict())

    return match_set._action_sets.get(action, make_empty_action_set())


def get_conditions_in_match_set(match_set: xcs.MatchSet) -> list[BitCondition]:
    all_rules = []
    for action in match_set:
        all_rules.extend(match_set[action])

    return all_rules


def get_rules_in_action_set(action_set: xcs.ActionSet) -> list[xcs.XCSClassifierRule]:
    return list(action_set._rules.values())



def rules_to_population(rules: list[xcs.XCSClassifierRule]) -> dict:
    return {rule.condition : {True : rule} for rule in rules}
