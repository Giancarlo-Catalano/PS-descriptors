import numpy as np
from sklearn.metrics import mean_squared_error

from Core.FullSolution import FullSolution
from Core.PRef import PRef


class AbstractDecisionTreeRegressor:
    maximum_depth: int
    def __init__(self, maximum_depth: int):
        self.maximum_depth = maximum_depth

    def __repr__(self):
        raise NotImplemented

    def train_from_pRef(self, pRef: PRef, random_state: int) -> None:
        raise NotImplemented

    def get_prediction(self, solution: FullSolution) -> float:
        raise NotImplemented

    def get_mse_on_test_data(self, test_pRef: PRef) -> float:
        evaluated_solutions = test_pRef.get_evaluated_FSs()
        predictions = np.array([self.get_prediction(solution) for solution in evaluated_solutions])
        actual_values = test_pRef.fitness_array

        return mean_squared_error(actual_values, predictions)

