import xcs
from xcs import ClassifierSet, MatchSet
from xcs.bitstrings import BitString

from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from LCS.Conversions import condition_to_ps


class SolutionDifferenceModel(ClassifierSet):
    verbose: bool

    def __init__(self,
                 algorithm,
                 possible_actions,
                 verbose=False
                 ):
        super().__init__(algorithm, possible_actions)
        self.verbose = verbose

    def update_match_set_after_covering(self, old_match_set: xcs.MatchSet,
                                        by_action: dict[bool, dict],
                                        new_rules: list) -> xcs.MatchSet:
        # Add the new classifier, getting back a list of the rule(s)
        # which had to be removed to make room for it.
        replaced = [removed for rule in new_rules for removed in self.add(rule)]  # MODIFIED
        if self.verbose and len(replaced) > 0:
            print("In adding those rules, the following were removed")
            for replaced_rule in replaced:
                print(replaced_rule)

        # Remove the rules that were removed the population from the
        # action set, as well. Note that they may not appear in the
        # action set, in which case nothing is done.
        for replaced_rule in replaced:  # MODIFIED
            condition = replaced_rule.condition
            if condition in by_action[True]:
                del by_action[True][condition]
            else:
                del by_action[False][condition]

        # Add the new classifier to the action set. This is done after
        # the replaced rules are removed, just in case the algorithm
        # provided us with a rule that was already present and was
        # displaced.
        for rule in new_rules:  # MODIFIED
            by_action[True][rule.condition] = rule

        # Reconstruct the match set with the modifications we just
        # made.
        by_action = self.remove_empty_entries_from_dict(by_action)  # silly library
        return MatchSet(self, old_match_set.situation, by_action)

    @classmethod
    def remove_empty_entries_from_dict(cls, old_dict):
        new_dict = old_dict.copy()
        for key in old_dict:
            if len(old_dict[key]) == 0:
                del new_dict[key]
        return new_dict

    def match(self, situation: (EvaluatedFS, EvaluatedFS)):
        # modified from the original because we might want to add many rules

        # Find the conditions that match against the current situation, and
        # group them according to which action(s) they recommend.
        by_action = {True: dict(), False: dict()}  # TODO explain the fact that by_action does not behave as normal

        winner, loser = situation

        for condition, actions in self._population.items():
            matches_winner = condition(winner.values)
            matches_loser = condition(loser.values)

            if matches_winner == matches_loser:  # we only want the cases where only one is matched
                continue

            for action, rule in actions.items():
                by_action[matches_winner][condition] = rule  # the action is whether it's in the winner or loser

        by_action = self.remove_empty_entries_from_dict(by_action)  # silly library
        # Construct the match set.
        match_set = MatchSet(self, situation, by_action)

        # Apply covering if necessary.
        if self._algorithm.covering_is_required(match_set):
            # Ask the algorithm to provide a new classifier rule to add to the population.
            rules: list = self._algorithm.cover_with_many(match_set)  # MODIFIED

            # Ensure that the condition provided by the algorithm does indeed match the situation.
            assert (all(rule.condition(winner.values) for rule in rules))

            return self.update_match_set_after_covering(old_match_set=match_set,
                                                        by_action=by_action,
                                                        new_rules=rules)

        # Return the newly created match set.
        return match_set

    def predict(self, sol_pair: (EvaluatedFS, EvaluatedFS)) -> dict:
        # This function will need to be expanded in the future
        # It mainly stems from my confusion in not finding a model.predict function .. does it really not exist???
        """ The result dict has the following entries
            prediction: the predicted class (0 = bad, 1 = good solution)
            rules: list of matched rules, ordered by 'prediction weight'
        """
        match_set = self.match(sol_pair)
        selected_action = match_set.select_action()
        rules = list(match_set[selected_action])

        def get_rule_quality(rule):
            return rule.prediction_weight

        rules.sort(key=get_rule_quality, reverse=True)  # the best are first

        result = {"prediction": selected_action,
                  "rules": rules}

        return result
