from Core.FullSolution import FullSolution
from Core.PRef import PRef
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor


class NaiveRegressorWrapper(AbstractDecisionTreeRegressor):
    regressor: DecisionTreeRegressor

    def __init__(self, maximum_depth: int):
        self.regressor = DecisionTreeRegressor(max_depth=maximum_depth)
        super().__init__(maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: int = 42) -> None:
        self.regressor.fit(X=pRef.full_solution_matrix, y=pRef.fitness_array)

    def get_prediction(self, solution: FullSolution) -> float:
        return self.regressor.predict(X=solution.values.reshape((1, -1)))[0]

    def __repr__(self):
        return "NaiveRegressorWrapper"

    def get_mse_on_test_data(self, test_pRef: PRef) -> float:
        predictions = self.regressor.predict(test_pRef.full_solution_matrix)
        actual_values = test_pRef.fitness_array

        return mean_squared_error(actual_values, predictions)





