from typing import Optional

import numpy as np

from Core.FullSolution import FullSolution
from Core.PRef import PRef
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from interpretableai import iai


class IAIDecisionTree(AbstractDecisionTreeRegressor):
    # just a simple wrapper over iai.OptimalTreeRegressor
    regressor: Optional[iai.Learner]

    def __init__(self, maximum_depth: int):
        self.regressor = None
        super().__init__(maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: int = 42) -> None:
        self.regressor = iai.GridSearch(
            iai.OptimalTreeRegressor(
                random_seed=random_state,
            ),
            max_depth=range(1, self.maximum_depth),
        )
        print(f"The pref has {pRef}, {pRef.full_solution_matrix.shape = }, {pRef.fitness_array.shape =}")
        self.regressor.fit(pRef.full_solution_matrix, pRef.fitness_array)
        self.regressor.get_learner()

    def get_prediction(self, solution: FullSolution) -> float:
        print(f"iaidt.get_prediction")
        return self.regressor.predict(X=solution.values.reshape((1, -1)))[0]


    def get_predictions(self, solution_matrix: np.ndarray) -> np.ndarray:
        print(f"iaidt.get_predictions({solution_matrix.shape = })")
        return self.regressor.predict(solution_matrix)

    def __repr__(self):
        return repr(self.regressor)
