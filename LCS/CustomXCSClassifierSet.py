from xcs import ClassifierSet, MatchSet
from xcs.bitstrings import BitString

from Core.FullSolution import FullSolution
from LCS.Conversions import condition_to_ps


class CustomXCSClassiferSet(ClassifierSet):
    """ this class overrides ClassifierSet so that when covering is required I can add more than one rule at a time"""
    """ (In XCS it seems that you can only generate one new rule per covering...)"""

    """ Other than that, the changes are minimal"""

    verbose: bool


    def __init__(self,
                 algorithm,
                 possible_actions,
                 verbose = False
                 ):
        super().__init__(algorithm, possible_actions)
        self.verbose = verbose



    def match(self, situation):
        # modified from the original because we might want to add many rules

        # Find the conditions that match against the current situation, and
        # group them according to which action(s) they recommend.
        by_action = {}
        for condition, actions in self._population.items():
            if not condition(situation):
                continue

            for action, rule in actions.items():
                if action in by_action:
                    by_action[action][condition] = rule
                else:
                    by_action[action] = {condition: rule}

        # Construct the match set.
        match_set = MatchSet(self, situation, by_action)

        # If an insufficient number of actions are recommended, create some
        # new rules (condition/action pairs) until there are enough actions
        # being recommended.
        if self._algorithm.covering_is_required(match_set):
            # Ask the algorithm to provide a new classifier rule to add to
            # the population.
            rules: list = self._algorithm.cover_with_many(match_set)  # MODIFIED

            # Ensure that the condition provided by the algorithm does
            # indeed match the situation. If not, there is a bug in the
            # algorithm.
            for rule in rules:   # MODIFIED
                assert rule.condition(situation)

            # Add the new classifier, getting back a list of the rule(s)
            # which had to be removed to make room for it.
            replaced = [removed for rule in rules for removed in self.add(rule) ]   # MODIFIED
            if self.verbose and len(replaced) > 0:
                print("In adding those rules, the following were removed")
                for replaced_rule in replaced:
                    print(replaced_rule)
                    # ps = condition_to_ps(replaced_rule.condition)
                    # action = replaced_rule.action
                    # print(f"---->{ps} -> {action}")


            # Remove the rules that were removed the population from the
            # action set, as well. Note that they may not appear in the
            # action set, in which case nothing is done.
            for replaced_rule in replaced:  # MODIFIED
                action = replaced_rule.action
                condition = replaced_rule.condition
                if action in by_action and condition in by_action[action]:
                    del by_action[action][condition]
                    if not by_action[action]:
                        del by_action[action]

            # Add the new classifier to the action set. This is done after
            # the replaced rules are removed, just in case the algorithm
            # provided us with a rule that was already present and was
            # displaced.
            for rule in rules:  # MODIFIED
                if rule.action not in by_action:
                    by_action[rule.action] = {}
                by_action[rule.action][rule.condition] = rule

            # Reconstruct the match set with the modifications we just
            # made.
            match_set = MatchSet(self, situation, by_action)

        # Return the newly created match set.
        return match_set


    def predict(self, solution: FullSolution) -> dict:
        # This function will need to be expanded in the future
        # It mainly stems from my confusion in not finding a model.predict function .. does it really not exist???
        """ The result dict has the following entries
            prediction: the predicted class (0 = bad, 1 = good solution)
            rules: list of matched rules, ordered by 'prediction weight'
        """
        as_input = BitString(solution.values)
        match_set = self.match(as_input)
        selected_action = match_set.select_action()
        rules = list(match_set[selected_action])

        def get_rule_quality(rule):
            return rule.prediction_weight

        rules.sort(key=get_rule_quality, reverse=True)  # the best are first

        result = {"prediction": selected_action,
                  "rules": rules}

        return result