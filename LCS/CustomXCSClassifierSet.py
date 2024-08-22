from xcs import ClassifierSet, MatchSet


class CustomXCSClassiferSet(ClassifierSet):

    def __init__(self,algorithm, possible_actions):
        super().__init__(algorithm, possible_actions)



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
            rules: list = self._algorithm.cover_with_many(match_set)

            # Ensure that the condition provided by the algorithm does
            # indeed match the situation. If not, there is a bug in the
            # algorithm.
            for rule in rules:
                assert rule.condition(situation)

            # Add the new classifier, getting back a list of the rule(s)
            # which had to be removed to make room for it.
            replaced = [removed for rule in rules for removed in self.add(rule) ]

            # Remove the rules that were removed the population from the
            # action set, as well. Note that they may not appear in the
            # action set, in which case nothing is done.
            for replaced_rule in replaced:
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
            for rule in rules:
                if rule.action not in by_action:
                    by_action[rule.action] = {}
                by_action[rule.action][rule.condition] = rule

            # Reconstruct the match set with the modifications we just
            # made.
            match_set = MatchSet(self, situation, by_action)

        # Return the newly created match set.
        return match_set