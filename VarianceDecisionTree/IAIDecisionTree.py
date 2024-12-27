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
    use_hyperplanes: bool

    def __init__(self, maximum_depth: int,
                 use_hyperplanes: bool = False):
        self.regressor = None
        self.use_hyperplanes = use_hyperplanes
        super().__init__(maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: int = 42) -> None:
        if self.use_hyperplanes:
            self.regressor = iai.GridSearch(
                iai.OptimalTreeRegressor(
                    random_seed=random_state, # does the hyperplane config go here?
                ),
                max_depth=range(1, self.maximum_depth),
            )
        else:
            self.regressor = iai.GridSearch(
                iai.OptimalTreeRegressor(
                    random_seed=random_state,
                ),
                max_depth=range(1, self.maximum_depth),
            )
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

    def __repr__(self):
        return repr(self.regressor)
