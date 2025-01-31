from typing import Optional, Any, Callable

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from FSStochasticSearch.Operators import FSMutationOperator, FSCrossoverOperator
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.recursive_pRef_splitting import split_pRef_using_ps

class PSDecisionTree(AbstractDecisionTreeRegressor):

    # as a branching node
    split_ps: Optional[PS]
    unmatching_branch: Optional[Any]  # PSDecisionTree
    matching_branch: Optional[Any]  # PSDecisionTree

    # prediction
    own_average: Optional[float]

    # to pretty_print

    ps_budget: int
    ps_search_population_size: int




    own_variance: Optional[float]

    ancestor_splits: list[PS]
    optimisation_problem: BenchmarkProblem

    repr_ps: Callable

    avoid_ancestors: bool
    metrics_to_use: str

    def __init__(self,
                 maximum_depth: int,
                 ps_budget: int,
                 ps_search_population_size: int,
                 problem: BenchmarkProblem,
                 metrics_to_use: str,
                 ancestor_splits: Optional[list[PS]] = None,
                 avoid_ancestors: bool = False
                 ):
        self.ps_budget = ps_budget
        self.ps_search_population_size = ps_search_population_size
        self.split_ps = None
        self.unmatching_branch = None
        self.matching_branch = None

        self.optimisation_problem = problem

        self.ancestor_splits = [] if ancestor_splits is None else ancestor_splits

        self.own_variance = None
        self.own_average = None

        self.avoid_ancestors = avoid_ancestors
        self.metrics_to_use = metrics_to_use
        super().__init__(maximum_depth)


        self.repr_ps = repr


    def __repr__(self):
        return "PSDecisionTree"


    def is_leaf(self) -> bool:
        return self.split_ps is None

    def is_branch(self) -> bool:
        return not self.is_leaf()


    def set_repr_ps(self, repr_ps):
        self.repr_ps = repr_ps
        if self.is_branch():
            self.matching_branch.set_repr_ps(repr_ps)
            self.unmatching_branch.set_repr_ps(repr_ps)

    def train_from_pRef(self, pRef: PRef, random_state: int = 42, verbose = False) -> None:
        # every node has some statistics
        pRef_variance = float(np.var(pRef.fitness_array))
        self.own_variance = pRef_variance
        self.own_average = np.average(pRef.fitness_array)
        self.own_sd = np.std(pRef.fitness_array)
        self.mean_error = np.average(np.abs(pRef.fitness_array - self.own_average))
        if (self.maximum_depth < 1) or (pRef_variance < 1e-05):
            return

        best_solution = pRef.get_best_solution()
        unexplained_vars = get_unexplained_parts(best_solution, self.ancestor_splits)

        with utils.announce(f"Searching for a ps in a branch with {pRef.sample_size} datapoints", verbose):
            pss = find_ps_in_solution(pRef=pRef,
                                      ps_budget=self.ps_budget,
                                      culling_method="elbow",
                                      population_size=self.ps_search_population_size,
                                      to_explain=best_solution,
                                      unexplained_mask=unexplained_vars,
                                      proportion_unexplained_that_needs_used=0,
                                      proportion_used_that_should_be_unexplained=0.9 if self.avoid_ancestors else 0,
                                      problem = self.optimisation_problem,
                                      metrics = self.metrics_to_use,
                                      verbose=False)

        self.split_ps = pss[0]
        print(f"The chosen ps is {self.split_ps}, it has order {self.split_ps.fixed_count()}")
        match_pRef, unmatch_pRef = split_pRef_using_ps(pRef, self.split_ps)

        self.matching_branch = PSDecisionTree(maximum_depth=self.maximum_depth - 1,
                                              ps_budget=self.ps_budget,
                                              ps_search_population_size=self.ps_search_population_size,
                                              ancestor_splits=self.ancestor_splits + [self.split_ps],
                                              problem = self.optimisation_problem,
                                              metrics_to_use=self.metrics_to_use)

        self.unmatching_branch = PSDecisionTree(maximum_depth=self.maximum_depth - 1,
                                                ps_budget=self.ps_budget,
                                                ps_search_population_size=self.ps_search_population_size,
                                                ancestor_splits=self.ancestor_splits,
                                                problem = self.optimisation_problem,
                                                metrics_to_use=self.metrics_to_use)

        # sue me
        self.matching_branch.train_from_pRef(match_pRef, random_state, verbose)
        self.unmatching_branch.train_from_pRef(unmatch_pRef, random_state, verbose)

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
            return f"Leaf(Average = {self.own_average:.2f}, sd = {self.own_sd:2f}, ae = {self.mean_error:.2f} variance = {self.own_variance:.2f})"
        else:
            head_repr = f"Split by {self.repr_ps(self.split_ps)}"
            matches_repr = "(matches) " + (self.matching_branch.repr_long())
            unmatches_repr = "(NOT matches)"  + (self.unmatching_branch.repr_long())
            return (f"{head_repr}"
                    f"\n{utils.indent(matches_repr)}"
                    f"\n{utils.indent(unmatches_repr)}")

    def get_orders(self):
        if self.split_ps is None:
            return dict()
        else:
            return {"own": self.split_ps.fixed_count(),
                    "matching": self.matching_branch.get_orders(),
                    "unmatching": self.unmatching_branch.get_orders()}


    @classmethod
    def construct_from_dict(cls, data_dict: dict):
        def read_ps(input_str: str) -> PS:
            values = [STAR if char == "*" else int(char) for char in input_str.split()]
            return PS(values)

        new_branch = PSDecisionTree(maximum_depth=None,
                                  ps_budget=None,
                                  ps_search_population_size=None,
                                  problem = None,
                                  metrics_to_use = None
                                  )
        if "split_ps" in data_dict:
            split_ps = read_ps(data_dict["split_ps"])
            matching_branch = cls.construct_from_dict(data_dict["match"])
            non_matching_branch = cls.construct_from_dict(data_dict["not_match"])
            new_branch.split_ps = split_ps
            new_branch.matching_branch = matching_branch
            new_branch.unmatching_branch = non_matching_branch

        average = data_dict["average"]
        sd = data_dict["sd"]
        mean_error = data_dict["mean_error"]
        variance = data_dict["variance"]



        new_branch.own_average = average
        new_branch.own_sd = sd
        new_branch.mean_error = mean_error
        new_branch.own_variance = variance
        return new_branch

    def as_dict(self) -> dict:
        def write_ps(ps: PS) -> str:
            return " ".join("*" if value == STAR else value for value in ps.values)


        result = dict()
        if self.split_ps is not None:
            result["split_ps"] = write_ps(self.split_ps)
            result["matching_branch"] = self.matching_branch.as_dict()
            result["unmatching_branch"] = self.unmatching_branch.as_dict()

        result["average"] = self.own_average
        result["sd"] = self.own_sd
        result["mean_error"] = self.mean_error
        result["variance"] = self.own_variance
        return result

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


    def get_orders(self) -> dict:
        return self.original_dt.get_orders()

