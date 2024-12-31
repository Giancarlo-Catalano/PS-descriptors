from typing import Optional

import numpy as np
from scipy.stats import t, wilcoxon

from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Core.PSMetric.Metric import Metric
from scipy.stats import mannwhitneyu

from Core.SearchSpace import SearchSpace


class SignificantlyHighAverage(Metric):
    pRef: Optional[PRef]
    pRef_mean: Optional[float]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.pRef_mean = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.pRef_mean = np.average(self.pRef.fitness_array)

    def __repr__(self):
        return "Significance of Core"

    def get_p_value_and_sample_mean(self, ps: PS) -> (float, float):
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return -1, -1

        t_score = (sample_mean - self.pRef_mean) / (sample_stdev / np.sqrt(n))
        p_value = 1 - t.cdf(abs(t_score), df=n - 1)
        return p_value, sample_mean

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return 0

        t_score = (sample_mean - self.normalised_population_mean) / (sample_stdev / np.sqrt(n))
        cumulative_score = t.cdf(abs(t_score), df=n - 1)  # p_value = 1 - cumulative_score

        return 1 - cumulative_score

        # def invert_and_augment(score: float):
        #     return 1. - np.sqrt(score * (2. - score))
        #     # return 1. - np.sqrt(1-np.square(score))
        #
        # return invert_and_augment(cumulative_score)


class MannWhitneyU(Metric):
    pRef: Optional[PRef]
    test_method: str

    def __init__(self):
        self.pRef = None
        self.test_method = "asymptotic"

        super().__init__()

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def get_p_value(self, first_group: np.ndarray, second_group: np.ndarray) -> float:
        if min(len(first_group), len(second_group)) < 3:
            return 1.0
        test = mannwhitneyu(first_group, second_group, alternative="two-sided", method=self.test_method)
        return test.pvalue

    def get_p_value_fast(self, first_group: np.ndarray, second_group: np.ndarray, restrict_size: int) -> float:

        test = mannwhitneyu(first_group[:restrict_size], second_group[:restrict_size], alternative="two-sided",
                            method=self.test_method)
        return test.pvalue

    def test_effect(self, ps: PS) -> float:
        when_present, when_absent = self.pRef.fitnesses_of_observations_and_complement(ps)
        if len(when_present) < 2 or len(when_absent) < 2:
            return 1

        return self.get_p_value_fast(first_group=when_present, second_group=when_absent, restrict_size=1000)
        # return self.get_p_value(first_group=when_present, second_group=when_absent)

    def get_single_score(self, ps: PS) -> float:
        """This is not meant to be used but I might as well write this one line"""
        if ps.is_empty():
            return 1
        return self.test_effect(ps)


class WilcoxonTest:
    """ This is not to be used as a fitness function!!!"""
    sample_size: int
    search_space: SearchSpace

    fitness_evaluator: FSEvaluator

    def __init__(self,
                 sample_size: int,
                 search_space: SearchSpace,
                 fitness_evaluator: FSEvaluator):
        self.sample_size = sample_size
        self.search_space = search_space
        self.fitness_evaluator = fitness_evaluator
        super().__init__()

    def get_random_samples_without_pattern(self, ps: PS) -> list[FullSolution]:
        if ps.is_empty():
            raise Exception("Attempted to test for the empty pattern")

        result = []
        while len(result) < self.sample_size:
            candidate = FullSolution.random(self.search_space)
            if not contains(candidate, ps):
                result.append(candidate)

        return result

    def apply_pattern_to_samples(self, samples: list[FullSolution], ps: PS) -> list[FullSolution]:
        samples_matrix = np.array([sample.values.copy() for sample in samples])
        samples_matrix[:, ps.values != STAR] = ps.values[ps.values != STAR]

        return [FullSolution(row) for row in samples_matrix]

    def get_fitnesses(self, unevaluated_solutions: list[FullSolution]) -> list[float]:
        return [self.fitness_evaluator.evaluate(solution)
                for solution in unevaluated_solutions]

    @classmethod
    def get_p_values_given_fitnesses(cls, fitnesses_without: np.ndarray, fitnesses_with: np.ndarray) -> (float, float):
        """ returns p_value for greater, lower"""
        differences = fitnesses_with - fitnesses_without
        res_greater = wilcoxon(differences, alternative="greater")
        res_lower = wilcoxon(differences, alternative="less")
        return (float(res_greater.pvalue), float(res_lower.pvalue))

    def get_p_values_of_ps(self, ps: PS) -> (float, float):
        """ returns p_value for greater, lower"""
        without_pattern = self.get_random_samples_without_pattern(ps)
        with_pattern = self.apply_pattern_to_samples(without_pattern, ps)

        fitnesses_without = np.array(self.get_fitnesses(without_pattern))
        fitnesses_with = np.array(self.get_fitnesses(with_pattern))

        return self.get_p_values_given_fitnesses(fitnesses_without, fitnesses_with)


class WilcoxonNearOptima:
    evaluator: FSEvaluator
    sorted_pRef: PRef
    samples_required: int

    def __init__(self,
                 evaluator: FSEvaluator,
                 pRef: PRef,
                 samples_required: int):
        self.evaluator = evaluator
        self.sorted_pRef = pRef.get_sorted()
        self.samples_required = samples_required

    def get_solutions_without_pattern(self, ps: PS) -> (np.ndarray, np.ndarray):
        """returns the solutions and their original fitness"""
        pattern = ps.values

        rows_to_be_observed = self.sorted_pRef.full_solution_matrix[:, pattern != STAR]
        pattern_fixed_values = pattern[pattern != STAR]

        winning_rows_indexes = []
        for row_index, observed_row in enumerate(rows_to_be_observed):
            if not np.array_equal(observed_row, pattern_fixed_values):
                winning_rows_indexes.append(row_index)
                if len(winning_rows_indexes) >= self.samples_required:
                    break
        else:
            raise Exception(
                f"Was not able to gather enough solutions without the pattern {ps} to prove that it is good")

        full_solution_matrix = self.sorted_pRef.full_solution_matrix[winning_rows_indexes]
        fitnesses = self.sorted_pRef.fitness_array[winning_rows_indexes]
        return full_solution_matrix, fitnesses

    def get_new_fitnesses_when_forcing_ps(self, old_solutions: np.ndarray, ps: PS) -> np.ndarray:
        samples_matrix = old_solutions.copy()
        samples_matrix[:, ps.values != STAR] = ps.values[ps.values != STAR]
        fitnesses = np.array([self.evaluator.evaluate(FullSolution(row)) for row in samples_matrix])

        return fitnesses

    def get_p_values_of_ps(self, ps: PS) -> (float, float):
        solutions_without, fitnesses_without = self.get_solutions_without_pattern(ps)
        fitnesses_with = self.get_new_fitnesses_when_forcing_ps(solutions_without, ps)
        return WilcoxonTest.get_p_values_given_fitnesses(fitnesses_without, fitnesses_with)


def effect_string(greater_p, lower_p, threshold=0.05) -> str:
    greater_is_significant = greater_p < threshold
    lower_is_significant = lower_p < threshold

    match (greater_is_significant, lower_is_significant):
        case (True, False):
            return "Positive"
        case (False, True):
            return "Negative"
        case (True, True):
            return "Both"
        case (False, False):
            return "Inconclusive"

def get_hypothesis_string(ps: PS,
                          hypothesis_tester: WilcoxonTest,
                          near_optima_hypothesis_tester: WilcoxonNearOptima):
    greater_p_value, lower_p_value = hypothesis_tester.get_p_values_of_ps(ps)
    near_optima_greater_p_value, near_optima_lower_p_value = near_optima_hypothesis_tester.get_p_values_of_ps(ps)

    threshold = 0.05

    human_string = (f"Global effect: {effect_string(greater_p_value, lower_p_value)}, "
                    f"local optima effect: {effect_string(near_optima_greater_p_value, near_optima_lower_p_value)}")

    p_values_string = (f"{greater_p_value = }, "
                       f"{lower_p_value = }, "
                       f"{near_optima_greater_p_value = }, "
                       f"{near_optima_lower_p_value = }")

    return human_string + "\n" + p_values_string
