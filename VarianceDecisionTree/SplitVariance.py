import numpy as np

from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats


class SplitVariance(Metric):
    selected_pRef: PRef

    def __init__(self,
                 selected_pRef: PRef):
        self.selected_pRef = selected_pRef
        super().__init__()

    @classmethod
    def get_split_indexes_of_ps(cls, pRef, ps: PS) -> (np.ndarray, np.ndarray):
        matching_indexes = pRef.get_indexes_matching_ps(ps)
        not_matches = np.ones(shape=pRef.fitness_array.shape, dtype=bool)
        not_matches[matching_indexes] = False
        not_matching_indexes = np.arange(pRef.sample_size)[not_matches]
        return matching_indexes, not_matching_indexes

    @classmethod
    def get_weighted_variance(cls, values_a: np.ndarray, values_b : np.ndarray):
        len_a = len(values_a)
        len_b = len(values_b)

        match (len_a < 2, len_b < 2):
            case (True, True):
                raise Exception("The matches and complements are both too small")
            case (True, False):
                return np.var(values_b)
            case (False, True):
                return np.var(values_a)
            case (False, False):
                return (len_a/(len_a+len_b)) * np.var(values_a) + (len_b/(len_a+len_b)) * np.var(values_b) # weighted variance

    def get_single_score(self, ps: PS) -> (ArrayOfFloats,ArrayOfFloats):
        matching_fitnesses, not_matching_fitnesses = self.selected_pRef.fitnesses_of_observations_and_complement(ps)
        return self.get_weighted_variance(matching_fitnesses, not_matching_fitnesses)

