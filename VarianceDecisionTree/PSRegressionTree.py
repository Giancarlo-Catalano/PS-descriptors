from dataclasses import dataclass
from typing import Optional, Iterator, Callable

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Explanation.MinedPSManager import MinedPSManager
from Explanation.PRefManager import PRefManager
from Explanation.PSPropertyManager import PSPropertyManager
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from VarianceDecisionTree.PSDecisionTree import PSDecisionTree
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution


@dataclass
class PSSearchSettings:
    ps_search_budget: int
    ps_search_population: int
    metrics: str
    avoid_ancestors: bool

    culling_method: str
    original_problem: Optional[BenchmarkProblem]
    verbose: bool

    def as_dict(self):
        return {"ps_search_budget": self.ps_search_budget,
                "ps_search_population": self.ps_search_population,
                "metrics": self.metrics,
                "avoid_ancestors": self.avoid_ancestors,
                "culling_method": self.culling_method,
                "verbose": self.verbose}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(ps_search_budget=d["ps_search_budget"],
                   ps_search_population=d["ps_search_population"],
                   metrics = d["metrics"],
                   avoid_ancestors= d["avoid_ancestors"],
                   culling_method= d["culling_method"],
                   original_problem=None,
                   verbose=d["verbose"])


class PSRegressionTreeNode:
    prediction: float

    other_statistics: dict[str, float]

    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float]):
        self.prediction = prediction
        self.other_statistics = other_statistics

    @classmethod
    def get_statistics_from_pRef(cls, pRef: PRef) -> dict[str, float]:
        fitnesses = pRef.fitness_array

        stats = dict()
        stats["n"] = len(fitnesses)

        if len(fitnesses) > 0:
            stats["average"] = np.average(fitnesses)

        if len(fitnesses) > 1:
            stats["variance"] = np.var(fitnesses)
            stats["sd"] = np.std(fitnesses)
            average = stats["average"]
            stats["mse"] = np.average((fitnesses - average) ** 2)
            stats["mae"] = np.average(np.abs(fitnesses - average))

        return stats

    def as_dict(self):
        result = self.other_statistics.copy()
        result["prediction"] = self.prediction
        return result

    @classmethod
    def from_pRef(cls, pRef: PRef):
        stats = PSRegressionTreeLeafNode.get_statistics_from_pRef(pRef)
        prediction = stats.get("average", float('nan'))
        return cls(prediction=prediction, other_statistics=stats)

    def repr_custom(self, custom_ps_repr) -> str:
        raise NotImplemented

    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplemented


class PSRegressionTreeLeafNode(PSRegressionTreeNode):
    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float]):
        super().__init__(prediction=prediction, other_statistics=other_statistics)

    def __repr__(self):
        return f"LeafNode(prediction = {self.prediction:.2f}, mae = {self.other_statistics['mae']:.2f})"

    def repr_custom(self, custom_ps_repr: Callable):
        return self.__repr__()

    def as_dict(self) -> dict:
        return {"node_type": "leaf"} | super().as_dict()

    @classmethod
    def from_dict(cls, d:dict):
        assert(d["node_type"] == "leaf")
        other_statistics = d.copy()
        del other_statistics["prediction"]
        del other_statistics["node_type"]
        return cls(prediction=d["prediction"],
                   other_statistics=other_statistics)


class PSRegressionTreeBranchNode(PSRegressionTreeNode):
    split_ps: Optional[PS]
    ps_properties: Optional[list[(str, float, float)]]

    matching_branch: Optional
    not_matching_branch: Optional

    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float]):
        super().__init__(prediction=prediction, other_statistics=other_statistics)
        self.split_ps = None
        self.ps_properties = None

        self.matching_branch = None
        self.not_matching_branch = None

    @classmethod
    def find_splitting_ps(cls,
                          search_settings: PSSearchSettings,
                          pRef: PRef,
                          ancestors: Optional[list[PS]]) -> PS:
        best_solution = pRef.get_best_solution()
        unexplained_vars = get_unexplained_parts(best_solution, [] if ancestors is None else ancestors)

        ps_candidates = find_ps_in_solution(pRef=pRef,
                                            ps_budget=search_settings.ps_search_budget,
                                            culling_method=search_settings.culling_method,
                                            population_size=search_settings.ps_search_population,
                                            to_explain=best_solution,
                                            unexplained_mask=unexplained_vars,
                                            proportion_unexplained_that_needs_used=0,
                                            proportion_used_that_should_be_unexplained=0.8 if search_settings.avoid_ancestors else 0,
                                            problem=search_settings.original_problem,
                                            metrics=search_settings.metrics,
                                            verbose=search_settings.verbose)

        return ps_candidates[0]

    def repr_custom(self, custom_ps_repr: Callable):
        ps_repr: str = custom_ps_repr(self.split_ps)
        is_multiline = len(ps_repr.split("\n")) > 1

        result = ""
        if is_multiline:
            result = (f"Branching, split ps = \n"
                      f"{ps_repr}\n"
                      f"prediction = {self.prediction:.2f}, mae = {self.other_statistics['mae']:.2f})")
        else:
            result = (f"Branching, split ps = {ps_repr}, "
                      f"prediction = {self.prediction:.2f}, "
                      f"mae = {self.other_statistics['mae']:.2f}")

        result += (f",\n"
                   f"matching = \n"
                   f"{utils.indent(self.matching_branch.repr_custom(custom_ps_repr))},\n"
                   f"not_matching = \n"
                   f"   {utils.indent(self.not_matching_branch.repr_custom(custom_ps_repr))}")

        return result

    def as_dict(self) -> dict:
        own_dict = {"node_type": "branch",
                    "split_ps": self.split_ps.__repr__(),
                    "matching_branch": self.matching_branch.as_dict(),
                    "not_matching_branch": self.not_matching_branch.as_dict()}

        return own_dict | super().as_dict()


    @classmethod
    def get_node_from_dict(cls, d: dict) -> PSRegressionTreeNode:
        if d["node_type"] == "branch":
            return cls.from_dict(d)
        else:
            return PSRegressionTreeLeafNode.from_dict(d)


    @classmethod
    def from_dict(cls, d: dict):
        assert(d["node_type"] == "branch")
        other_statistics = d.copy()
        del other_statistics["prediction"]
        del other_statistics["node_type"]
        result_node = cls(prediction=d["prediction"],
                   other_statistics=other_statistics)
        result_node.matching_branch = cls.get_node_from_dict(d["matching_branch"])
        result_node.not_matching_branch = cls.get_node_from_dict(d["not_matching_branch"])
        result_node.split_ps = PS(STAR if c == "*" else int(c) for c in d["split_ps"].split())
        return result_node


class PSRegressionTree(AbstractDecisionTreeRegressor):
    root_node: Optional[PSRegressionTreeNode]
    search_settings: Optional[PSSearchSettings]
    problem: Optional[BenchmarkProblem]

    def __init__(self, maximum_depth: int):
        self.root_node = None
        self.search_settings = None

        super().__init__(maximum_depth=maximum_depth)

    def train_from_pRef(self, pRef: PRef, random_state: int):

        def recursively_train_node(pRef_to_split: PRef,
                                   current_depth: int,
                                   ancestors: list[PS]) -> PSRegressionTreeNode:
            print(f"Splitting a pRef of size {pRef.sample_size}")
            if (current_depth >= self.maximum_depth) or (pRef.sample_size < 2):
                return PSRegressionTreeLeafNode.from_pRef(pRef_to_split)

            # otherwise we split more
            node = PSRegressionTreeBranchNode.from_pRef(pRef_to_split)
            splitting_ps = PSRegressionTreeBranchNode.find_splitting_ps(search_settings=self.search_settings,
                                                                        pRef=pRef_to_split,
                                                                        ancestors=ancestors)

            if self.search_settings.verbose:
                print(f"The splitting PS is {splitting_ps}")
            node.split_ps = splitting_ps
            matching_indexes = pRef.get_indexes_matching_ps(splitting_ps)
            matching_pRef, not_matching_pRef = pRef.split_by_indexes(matching_indexes)

            node.matching_branch = recursively_train_node(pRef_to_split=matching_pRef,
                                                          current_depth=current_depth + 1,
                                                          ancestors=ancestors + [splitting_ps])

            node.not_matching_branch = recursively_train_node(pRef_to_split=not_matching_pRef,
                                                              current_depth=current_depth + 1,
                                                              ancestors=ancestors)

            return node

        self.root_node = recursively_train_node(pRef_to_split=pRef,
                                                current_depth=0,
                                                ancestors=[])

    def get_prediction(self, solution: FullSolution) -> float:

        def recursive_prediction(current_node: PSRegressionTreeNode) -> float:
            if isinstance(current_node, PSRegressionTreeLeafNode):
                return current_node.prediction
            elif isinstance(current_node, PSRegressionTreeBranchNode):
                if contains(solution, current_node.split_ps):
                    return recursive_prediction(current_node.matching_branch)
                else:
                    return recursive_prediction(current_node.not_matching_branch)

        return recursive_prediction(self.root_node)

    def all_nodes_as_list(self) -> list[PSRegressionTreeNode]:

        accumulator = []

        def recursively_register_node(current_node: PSRegressionTreeNode) -> None:
            accumulator.append(current_node)
            if isinstance(current_node, PSRegressionTreeBranchNode):
                recursively_register_node(current_node.matching_branch)
                recursively_register_node(current_node.not_matching_branch)

        recursively_register_node(self.root_node)
        return accumulator

    def add_properties_to_pss(self, ps_property_manager: PSPropertyManager):
        nodes_to_modify = [node for node in self.all_nodes_as_list() if isinstance(node, PSRegressionTreeBranchNode)]

        for node in nodes_to_modify:
            descriptors = ps_property_manager.get_significant_properties_of_ps(ps=node.split_ps)
            descriptors = ps_property_manager.sort_pvrs_by_rank(descriptors)
            node.ps_properties = descriptors#

    def all_pss_as_list(self) -> list[PS]:
        return [node.split_ps for node in self.all_nodes_as_list() if isinstance(node, PSRegressionTreeBranchNode)]

    def __repr__(self):
        if self.problem is None:
            repr_ps = repr
        else:
            repr_ps = self.problem.repr_ps
        if self.root_node is None:
            return "Invalid Tree"

        return self.root_node.repr_custom(repr_ps)


    def as_dict(self) -> dict:
        result = {"maximum_depth": self.maximum_depth}
        if self.search_settings is not None:
            result["search_settings"] = self.search_settings.as_dict()

        if self.root_node is not None:
            result["tree"] = self.root_node.as_dict()

        return result

    @classmethod
    def from_dict(cls, d: dict):
        result = cls(maximum_depth=d["maximum_depth"])
        result.root_node = PSRegressionTreeBranchNode.get_node_from_dict(d["tree"]) if "tree" in d else None
        return result

def test_ps_regression_tree():
    problem = RoyalRoad(5)

    with utils.announce("generating the pRef"):
        pRef = PRefManager.generate_pRef(problem=problem,
                                         sample_size=10000,
                                         which_algorithm="GA")

    search_settings = PSSearchSettings(ps_search_budget=2000,
                                       ps_search_population=50,
                                       metrics="simplicity variance ground_truth_atomicity",
                                       avoid_ancestors=True,
                                       original_problem=problem,
                                       culling_method="biggest",
                                       verbose=True)

    decision_tree = PSRegressionTree(maximum_depth=3)
    decision_tree.search_settings = search_settings

    with utils.announce("training the decision tree"):
        decision_tree.train_from_pRef(pRef, random_state=42)

    decision_tree.problem = problem
    print(decision_tree)

    mined_ps_manager = MinedPSManager(problem=problem,
                                           mined_ps_file=None,
                                           control_ps_file=None,
                                           verbose=False)

    mined_ps_manager.cached_pss = decision_tree.all_pss_as_list()
    mined_ps_manager.cached_control_pss = mined_ps_manager.generate_control_pss()

    ps_property_manager = PSPropertyManager(problem=problem,
                                                 property_table_file=None,
                                                 verbose=True,
                                                 threshold=0.1)

    ps_property_path = # TODO
    ps_property_manager.generate_property_table_file()

    ps_property_manager = PSPropertyManager(problem = problem,property_table_file=None, verbose = True)
    ps_property_manager.generate_property_table_file()



test_ps_regression_tree()
