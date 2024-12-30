import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

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


    def get_predictions(self, solution_matrix: np.ndarray) -> np.ndarray:
        # print(f"get_predictions({solution_matrix.shape = })")
        # this is the fallback if there isn't a more efficient way to do it
        return np.array([self.get_prediction(FullSolution(row)) for row in solution_matrix])

    def get_mse_on_test_data(self, test_pRef: PRef) -> float:
        #print(f"get_mse_on_test_data({test_pRef.full_solution_matrix.shape = })")
        predictions = self.get_predictions(test_pRef.full_solution_matrix)
        actual_values = test_pRef.fitness_array

        return mean_squared_error(actual_values, predictions)

    def get_error_metrics(self, test_pRef) -> dict:
        predictions = self.get_predictions(test_pRef.full_solution_matrix)
        ground_truth = test_pRef.fitness_array

        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        r_sq = r2_score(ground_truth, predictions)
        evs = explained_variance_score(ground_truth, predictions)

        return {"mse": mse,
                "mae": mae,
                "r_sq": r_sq,
                "evs": evs}

