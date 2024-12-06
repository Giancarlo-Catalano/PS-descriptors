from typing import Optional, Iterable

import numpy as np

from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Linkage.BivariateLinkage import BivariateLinkage
from Core.PSMetric.Metric import Metric
from VarianceDecisionTree.SplitVariance import SplitVariance


class VarianceSplitLinkage(BivariateLinkage, Metric):
    current_solution: FullSolution
    split_variance_metric: Optional[SplitVariance]
    linkage_dict: dict
    linkage_table: np.ndarray

    def __init__(self):
        self.split_variance_metric = None
        self.linkage_dict = None
        self.linkage_table = None
        super().__init__()

    def set_pRef(self, pRef: PRef) -> None:
        self.pRef = pRef
        self.split_variance_metric = SplitVariance(pRef)

        split_variance_dict = self.get_split_variance_dict()
        self.linkage_dict = self.get_linkage_dict(split_variance_dict)

    def get_split_variance_dict(self) -> dict:
        assert (self.pRef is not None)

        split_variance_dict = dict()
        # univariate_modifications
        for var, cardinality in enumerate(self.search_space.cardinalities):
            for val in range(cardinality):
                hyperplane = ((var, val),)
                variance = self.get_split_variance_of_hyperplane(hyperplane)
                split_variance_dict[hyperplane] = variance

        # bivariate modifications
        for var_a, val_a, var_b, val_b in self.every_var_val_pair_combination():
            hyperplane = ((var_a, val_a), (var_b, val_b))
            variance = self.get_split_variance_of_hyperplane(hyperplane)
            split_variance_dict[hyperplane] = variance

        return split_variance_dict

    def set_solution(self, new_solution: FullSolution) -> None:
        self.current_solution = new_solution
        self.linkage_table = self.get_linkage_table()

    def get_split_variance_of_hyperplane(self, hyperplane: Iterable) -> float:
        hyperplane_ps = PS.empty(self.search_space)
        for var, val in hyperplane:
            hyperplane_ps = hyperplane_ps.with_fixed_value(var, val)

        return self.split_variance_metric.get_single_score(hyperplane_ps)


    def every_var_val_combination(self) -> Iterable:
        return ((var, val)
                    for var, cardinality in enumerate(self.search_space.cardinalities)
                    for val in range(cardinality))

    def every_var_val_pair_combination(self) -> Iterable:
        return ((var_a, val_a, var_b, val_b)
                for var_a, var_b in self.every_var_pair_iterator()
                for val_a in range(self.search_space.cardinalities[var_a])
                for val_b in range(self.search_space.cardinalities[var_b]))

    def get_linkage_dict(self, variance_dict: dict) -> dict:
        def get_linkage(var_a, val_a, var_b, val_b) -> float:
            just_a = variance_dict[((var_a, val_a),)]
            just_b = variance_dict[((var_b, val_b),)]
            both = variance_dict[((var_a, val_a), (var_b, val_b))]
            return abs(both - just_a - just_b)  # although this could just be both

        def get_univariate_importance(var, val) -> float:
            return variance_dict[((var, val),)]

        bivariate_linkages = {((var_a, val_a), (var_b, val_b)): get_linkage(var_a, val_a, var_b, val_b)
                for var_a, val_a, var_b, val_b in self.every_var_val_pair_combination()}

        univariate_linkages = {((var, val),) : get_univariate_importance(var, val)
                               for var, val in self.every_var_val_combination()}

        return bivariate_linkages | univariate_linkages

    def get_linkage_table(self) -> np.ndarray:
        n = self.search_space.amount_of_parameters
        result = np.zeros(shape=(n, n), dtype=float)

        # bivariate info
        for var_a, var_b in self.every_var_pair_iterator():
            val_a = self.current_solution.values[var_a]
            val_b = self.current_solution.values[var_b]

            result[var_a, var_b] = self.linkage_dict[((var_a, val_a), (var_b, val_b))]

        result += result.T

        #univariate_importances = np.sum(result, axis=0) / ((self.n_vars - 1))
        univariate_importances = np.array([self.linkage_dict[((var, val),)]
                                           for var, val in enumerate(self.current_solution.values)])
        np.fill_diagonal(result, univariate_importances)

        return result

    def get_univariate_linkage_of_var(self, var: int) -> float:
        return self.linkage_table[var, var]

    def get_bivariate_linkage_between_vars(self, var_a: int, var_b: int) -> float:
        return self.linkage_table[var_a, var_b]