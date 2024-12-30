from typing import Optional

import numpy as np
import pandas as pd

from Core.FullSolution import FullSolution
from Core.PRef import PRef
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor

from interpretableai import iai



#todo
# decide the cp paramter for the tree
#  let it autodecide, set it to some special values etc
#  they call it the prescription_factor (cp stands for combined performance)
# use hyperplanes
#  hyperplane_config=(sparsity=:all,)
# use linear regression in the leaves
#       regression_features=All(),
#

def convert_numpy_array_to_df(array: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(array)
    df.astype("category")
    return df

class IAIDecisionTree(AbstractDecisionTreeRegressor):
    # just a simple wrapper over iai.OptimalTreeRegressor
    regressor: Optional[iai.Learner]

    prescription_factor: float
    use_hyperplanes: bool
    use_linear_regression_in_leaves: bool

    def __init__(self, maximum_depth: int,
                  prescription_factor: float = 0.5,
                 use_hyperplanes: bool = False,
                 use_linear_regression_in_leaves: bool = False):
        self.regressor = None
        self.prescription_factor = prescription_factor
        self.use_hyperplanes = use_hyperplanes
        self.use_linear_regression_in_leaves = use_linear_regression_in_leaves
        super().__init__(maximum_depth)


    def generate_untrained_regressor(self, random_state: int):
        if self.use_hyperplanes:
            return iai.GridSearch(
                    iai.OptimalTreeRegressor(
                        random_seed=random_state,
                        cp = self.prescription_factor,
                        hyperplane_config ={"sparsity": "all"},
                    ),
                    max_depth=range(1, self.maximum_depth),
                )
        else:
            return iai.GridSearch(
                iai.OptimalTreeRegressor(
                    random_seed=random_state,
                    cp=self.prescription_factor,
                ),
                max_depth=range(1, self.maximum_depth),
            )

    def __repr__(self):
        return "IAIDecisionTree"

    def train_from_pRef(self, pRef: PRef, random_state: int = 42) -> None:
        self.regressor = self.generate_untrained_regressor(random_state)
        # print(f"The pRef has {pRef}, {pRef.full_solution_matrix.shape = }, {pRef.fitness_array.shape =}")
        categorical_df = convert_numpy_array_to_df(pRef.full_solution_matrix)
        self.regressor.fit(categorical_df, pRef.fitness_array)
        self.regressor.get_learner()

    def get_prediction(self, solution: FullSolution) -> float:
        # print(f"iaidt.get_prediction")
        single_row = solution.values.reshape((1, -1))
        single_row = convert_numpy_array_to_df(single_row)
        return self.regressor.predict(X=single_row)[0]


    def get_predictions(self, solution_matrix: np.ndarray) -> np.ndarray:
        # print(f"iaidt.get_predictions({solution_matrix.shape = })")
        solution_df = convert_numpy_array_to_df(solution_matrix)
        return self.regressor.predict(solution_df)