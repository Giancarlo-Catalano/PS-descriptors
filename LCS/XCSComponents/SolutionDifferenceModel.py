import xcs
from tqdm import tqdm
from xcs import ClassifierSet, MatchSet, scenarios
from xcs.scenarios import ScenarioObserver

from Core.EvaluatedFS import EvaluatedFS
from LCS.Conversions import rules_to_population


class SolutionDifferenceModel(ClassifierSet):
    verbose: bool
    allow_ga_reproduction: bool
    want_negative_traits: bool

    def __init__(self,
                 algorithm,
                 want_negative_traits: bool,
                 possible_actions=(True, False),
                 allow_ga_reproduction: bool = True,
                 verbose=False
                 ):
        self.allow_ga_reproduction = allow_ga_reproduction
        self.want_negative_traits = want_negative_traits
        super().__init__(algorithm, possible_actions)
        self.verbose = verbose

    def add_rules_to_model_and_update_match_set_after_covering(self, old_match_set: xcs.MatchSet,
                                                               by_action: dict[bool, dict],
                                                               new_rules: list) -> xcs.MatchSet:
        # Add the rules to the model, and get the list of removed ones
        replaced = [removed for rule in new_rules for removed in self.add(rule)]  # MODIFIED
        if self.verbose:
            print("Adding the following rules")
            for added_rule in new_rules:
                print(added_rule.condition)
        if self.verbose and len(replaced) > 0:
            print("In adding those rules, the following were removed")
            for replaced_rule in replaced:
                print(replaced_rule.condition)

        # Remove the rules that were removed the population from the
        # action set, as well. Note that they may not appear in the
        # action set, in which case nothing is done.
        for replaced_rule in replaced:  # MODIFIED
            condition = replaced_rule.condition
            if condition in by_action[True]:
                del by_action[True][condition]
            elif condition in by_action[False]:
                del by_action[False][condition]
            else:
                if self.verbose and False:  # TODO Understand why this happens
                    print(f"The rule {replaced_rule} to be removed is nowhere to be found")

        # Add the new classifier to the action set. This is done after
        # the replaced rules are removed, just in case the algorithm
        # provided us with a rule that was already present and was
        # displaced.
        for rule in new_rules:  # MODIFIED
            by_action[True][rule.condition] = rule

        # Reconstruct the match set with the modifications we just made.
        # by_action = self.remove_empty_entries_from_dict(by_action)  # silly library
        return MatchSet(self, old_match_set.situation, by_action)

    @classmethod
    def remove_empty_entries_from_dict(cls, old_dict):
        """ This is dangerous, don't use it"""
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

        # by_action = self.remove_empty_entries_from_dict(by_action)  # silly library
        # Construct the match set.
        match_set = MatchSet(self, situation, by_action)

        # Apply covering if necessary.
        if self._algorithm.covering_is_required(match_set):
            # Ask the algorithm to provide a new classifier rule to add to the population.
            rules: list = self._algorithm.cover_with_many(match_set, only_return_one=True)  # MODIFIED TODO think if this is arbitrary

            # Ensure that the condition provided by the algorithm does indeed match the situation.
            assert (all(rule.condition(situation) for rule in rules))

            return self.add_rules_to_model_and_update_match_set_after_covering(old_match_set=match_set,
                                                                               by_action=by_action,
                                                                               new_rules=rules)

        # Return the newly created match set.
        return match_set


    def get_correct_set(self, match_set: MatchSet):
        return match_set[not self.want_negative_traits]

    def get_incorrect_set(self, match_set: MatchSet):
        return match_set[self.want_negative_traits]


    def apply_payoff_to_match_set(self, match_set: MatchSet):
        self._algorithm.apply_payoff_to_match_set(action_set=self.get_correct_set(match_set), payoff=1)
        self._algorithm.apply_payoff_to_match_set(action_set=self.get_incorrect_set(match_set), payoff=0)

    def use_match_set_for_learning(self, match_set: MatchSet):
        self.apply_payoff_to_match_set(match_set)
        if self._algorithm.allow_ga_reproduction:
            self._algorithm.introduce_rules_via_reproduction(action_set=self.get_correct_set(match_set))

    def run(self, scenario: ScenarioObserver, learn=True):
        """ Had to modify this since the match set is strange"""

        assert isinstance(scenario, scenarios.Scenario)

        # previous_match_set = None

        def single_iteration():
            # Gather information about the current state of the
            # environment.
            situation = scenario.sense()

            # Determine which rules match the current situation.
            match_set = self.match(situation)

            # No need to select an action
            # The reward for the rules that were betting on the correct side is 1.0,
            # and for those that bet against it, 0.0
            if learn:
                # apply payoff for correct instances
                self.apply_payoff_to_match_set(match_set)
                self._algorithm.update_match_set_timestamps(match_set)

                if self.algorithm.allow_ga_reproduction:
                    self._algorithm.introduce_rules_via_reproduction(action_set=match_set[True])
                match_set._closed = True  # vestigial but kept for completeness, check using assert(match_set.is_closed == False)

        for iteration in tqdm(range(scenario.wrapped.remaining_cycles)):
            if not scenario.more():
                break
            single_iteration()

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

    def set_rules(self, rules: list[xcs.XCSClassifierRule]):
        self._population = rules_to_population(rules)  # hopefully there are no cached values.
