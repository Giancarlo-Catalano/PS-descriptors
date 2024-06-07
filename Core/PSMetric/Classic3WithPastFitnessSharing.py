from typing import Iterable

import numpy as np

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.custom_types import ArrayOfFloats
from PSMiners.PyMoo.FitnessSharing import get_sharing_score_from_reference_group


class FitnessSharingPenaltyCalculator:
    current_archive_matrix: np.ndarray
    sigma_shared:float
    alpha: int


    def __init__(self, sigma_shared:float, alpha: int):
        self.current_archive_matrix = np.array([])
        self.sigma_shared = sigma_shared
        self.alpha = alpha

    @classmethod
    def list_of_pss_to_matrix(cls, pss: Iterable[PS]) -> np.ndarray:
        return np.array([ps.values for ps in pss])

    def set_current_archive(self, new_archive: list[PS]):
        self.current_archive_matrix = self.list_of_pss_to_matrix(new_archive)


    def calculate_penalty(self, ps: PS) -> float:
        return get_sharing_score_from_reference_group(self.current_archive_matrix,
                                                ps,
                                                sigma_shared=self.sigma_shared,
                                                alpha=self.alpha)

class Classic3WithPastFitnessSharing(Classic3PSEvaluator):
    fitness_penalty_calculator: FitnessSharingPenaltyCalculator


    def __init__(self, pRef: PRef):
        super().__init__(pRef = pRef)
        self.fitness_penalty_calculator = FitnessSharingPenaltyCalculator(0.5, 2)


    def set_archive(self, new_archive: list[PS]):
        self.fitness_penalty_calculator.set_current_archive(new_archive)


    def get_S_MF_A(self, ps: PS, invalid_value: float = 0) -> np.ndarray:
        unpenalised = super().get_S_MF_A(ps, invalid_value)
        penalty = self.fitness_penalty_calculator.calculate_penalty(ps)
        return unpenalised / (1+penalty)
