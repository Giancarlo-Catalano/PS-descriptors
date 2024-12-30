from typing import Optional, Any, Callable

import numpy as np

import utils
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains
from FSStochasticSearch.Operators import FSMutationOperator, FSCrossoverOperator
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.recursive_pRef_splitting import split_pRef_using_ps


class PSDecisionTree(AbstractDecisionTreeRegressor):
    ps_budget: int
    ps_search_population_size: int

    split_ps: Optional[PS]
    unmatching_branch: Optional[Any]  # PSDecisionTree
    matching_branch: Optional[Any]  # PSDecisionTree

    own_average: Optional[float]
    own_variance: Optional[float]

    ancestor_splits: list[PS]


    repr_ps: Callable

    def __init__(self,
                 maximum_depth: int,
                 ps_budget: int,
                 ps_search_population_size: int,
                 ancestor_splits: Optional[list[PS]] = None):
        self.ps_budget = ps_budget
        self.ps_search_population_size = ps_search_population_size
        self.split_ps = None
        self.unmatching_branch = None
        self.matching_branch = None

        self.ancestor_splits = [] if ancestor_splits is None else ancestor_splits

        self.own_variance = None
        self.own_average = None
        super().__init__(maximum_depth)


        self.repr_ps = repr


    def __repr__(self):
        return "PSDecisionTree"


    def set_repr_ps(self, repr_ps):
        self.repr_ps = repr_ps

    def train_from_pRef(self, pRef: PRef, random_state: int = 42) -> None:
        # print(f"Making a branch with max depth = {self.maximum_depth}, splitting a pref of size {pRef.sample_size}")
        pRef_variance = float(np.var(pRef.fitness_array))
        self.own_variance = pRef_variance
        self.own_average = np.average(pRef.fitness_array)
        if (self.maximum_depth < 1) or (pRef.sample_size < 20) or (pRef_variance < 1e-05):
            return

        best_solution = pRef.get_best_solution()
        unexplained_vars = get_unexplained_parts(best_solution, self.ancestor_splits)
        pss = find_ps_in_solution(pRef=pRef,
                                  ps_budget=self.ps_budget,
                                  culling_method="biggest",
                                  population_size=self.ps_search_population_size,
                                  to_explain=best_solution,
                                  unexplained_mask=unexplained_vars,
                                  proportion_unexplained_that_needs_used=0,
                                  proportion_used_that_should_be_unexplained=0,
                                  verbose=False)

        self.split_ps = pss[0]
        match_pRef, unmatch_pRef = split_pRef_using_ps(pRef, self.split_ps)

        self.matching_branch = PSDecisionTree(maximum_depth=self.maximum_depth - 1,
                                              ps_budget=self.ps_budget,
                                              ps_search_population_size=self.ps_search_population_size,
                                              ancestor_splits=self.ancestor_splits + [self.split_ps])

        self.unmatching_branch = PSDecisionTree(maximum_depth=self.maximum_depth - 1,
                                                ps_budget=self.ps_budget,
                                                ps_search_population_size=self.ps_search_population_size,
                                                ancestor_splits=self.ancestor_splits)

        # sue me
        self.matching_branch.train_from_pRef(match_pRef)
        self.unmatching_branch.train_from_pRef(unmatch_pRef)

    def get_prediction(self, solution: FullSolution) -> float:
        if self.split_ps is None:
            return self.own_average
        else:
            if contains(solution, self.split_ps):
                return self.matching_branch.get_prediction(solution)
            else:
                return self.unmatching_branch.get_prediction(solution)


    def get_prediction_with_restricted_depth(self, solution: FullSolution, allowed_depth: int) -> float:
        if allowed_depth == 0 or (self.split_ps is None):
            return self.own_average
        else:
            branch_to_navigate = self.matching_branch if contains(solution, self.split_ps) else self.unmatching_branch
            return branch_to_navigate.get_prediction_with_restricted_depth(solution, allowed_depth-1)

    def repr_long(self):
        if self.split_ps is None:
            return f"Leaf(Average = {self.own_average}"
        else:
            head_repr = f"Split by {self.repr_ps(self.split_ps)}"
            matches_repr = "(matches)" + repr(self.matching_branch)
            unmatches_repr = "(UNmatches)" + repr(self.unmatching_branch)
            return (f"{head_repr}"
                    f"\n{utils.indent(matches_repr)}"
                    f"\n{utils.indent(unmatches_repr)}")



class PSDecisionTreeRestrictedDepth(AbstractDecisionTreeRegressor):
    original_dt: PSDecisionTree
    depth: int

    def __init__(self, original_dt: PSDecisionTree,
                 depth: int):
        self.original_dt = original_dt
        self.depth = depth

        super().__init__(maximum_depth=depth)

    def get_prediction(self, solution: FullSolution) -> float:
        return self.original_dt.get_prediction_with_restricted_depth(solution, self.depth)

    def __repr__(self):
        return "PSDecisionTreeRestrictedDepth"

