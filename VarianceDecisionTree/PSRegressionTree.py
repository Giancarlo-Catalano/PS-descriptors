import json
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, contains, STAR
from Explanation.PSPropertyManager import PSPropertyManager
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from anytree import Node, RenderTree


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
                   metrics=d["metrics"],
                   avoid_ancestors=d["avoid_ancestors"],
                   culling_method=d["culling_method"],
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
            stats["min"] = np.min(fitnesses)
            stats["max"] = np.max(fitnesses)

        return stats

    def as_dict(self) -> dict:
        raise NotImplemented

    def get_prediction_dict(self) -> dict:
        prediction_dict = dict()
        prediction_dict["other_statistics"] = self.other_statistics
        prediction_dict["prediction"] = self.prediction
        return prediction_dict

    @classmethod
    def from_pRef(cls, pRef: PRef):
        stats = PSRegressionTreeLeafNode.get_statistics_from_pRef(pRef)
        prediction = stats.get("average", float('nan'))
        return cls(prediction=prediction, other_statistics=stats)

    def repr_custom(self, custom_ps_repr, custom_prop_repr) -> str:
        raise NotImplemented

    @classmethod
    def from_dict(cls, d: dict):
        raise NotImplemented


    def get_node_text(self, custom_repr_ps, custom_repr_properties) -> str:
        raise NotImplemented





class PSRegressionTreeLeafNode(PSRegressionTreeNode):
    def __init__(self,
                 prediction: float,
                 other_statistics: dict[str, float]):
        super().__init__(prediction=prediction, other_statistics=other_statistics)

    def __repr__(self):
        return f"LeafNode(prediction = {self.prediction:.2f}, mae = {self.other_statistics['mae']:.2f})"

    def repr_custom(self, custom_ps_repr: Callable, custom_prop_repr:Callable):
        return self.__repr__()

    def as_dict(self) -> dict:
        return {"node_type": "leaf"} | self.get_prediction_dict()

    @classmethod
    def from_dict(cls, d: dict):
        assert (d["node_type"] == "leaf")
        return cls(prediction=d["prediction"],
                   other_statistics=d["other_statistics"])

    def get_node_text(self, custom_ps_repr: Callable, custom_prop_repr: Callable):
        return f"(n = {self.other_statistics['n']}), prediction = {self.prediction:.2f} ± {self.other_statistics['mae']:.2f}, min = {self.other_statistics['min']:.2f}, max = {self.other_statistics['max']:.2f}\n"


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

        return ps_candidates[0] #  the culling method should leave just one left in the array anyway


    def get_node_text(self, custom_ps_repr: Callable, custom_prop_repr: Callable):
        ps_repr: str = custom_ps_repr(self.split_ps)
        is_multiline = len(ps_repr.split("\n")) > 1

        properties_str = "(no properties registered)"
        if self.ps_properties is not None:
            properties_str = custom_prop_repr(self.ps_properties)

        result = (f"(n = {self.other_statistics['n']}), prediction = {self.prediction:.2f} ± {self.other_statistics['mae']:.2f}, min = {self.other_statistics['min']:.2f}, max = {self.other_statistics['max']:.2f}\n"
                  f"Branching, split ps = {ps_repr}, \n"
                  f"properties: \n{utils.indent(properties_str)}")

        return result

    def repr_custom(self, custom_ps_repr: Callable, custom_prop_repr: Callable):
        result = self.get_node_text(custom_ps_repr, custom_prop_repr)

        result += (f",\n"
                   f"matching = \n"
                   f"{utils.indent(self.matching_branch.repr_custom(custom_ps_repr, custom_prop_repr))},\n"
                   f"not_matching = \n"
                   f"   {utils.indent(self.not_matching_branch.repr_custom(custom_ps_repr, custom_prop_repr))}")

        return result

    @classmethod
    def plain_ps_repr(cls, ps: PS) -> str:
        return " ".join("*" if value == STAR else f"{value}" for value in ps.values)


    def as_dict(self) -> dict:
        own_dict = {"node_type": "branch",
                    "split_ps": self.plain_ps_repr(self.split_ps),
                    "matching_branch": self.matching_branch.as_dict(),
                    "not_matching_branch": self.not_matching_branch.as_dict()}

        if self.ps_properties is not None:
            own_dict["ps_properties"] = self.ps_properties

        return own_dict | self.get_prediction_dict()

    @classmethod
    def get_node_from_dict(cls, d: dict) -> PSRegressionTreeNode:
        if d["node_type"] == "branch":
            return cls.from_dict(d)
        else:
            return PSRegressionTreeLeafNode.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict):
        assert (d["node_type"] == "branch")

        result_node = cls(prediction=d["prediction"],
                          other_statistics=d["other_statistics"])
        result_node.matching_branch = cls.get_node_from_dict(d["matching_branch"])
        result_node.not_matching_branch = cls.get_node_from_dict(d["not_matching_branch"])
        result_node.split_ps = PS(STAR if c == "*" else int(c) for c in d["split_ps"].split())
        result_node.ps_properties = d.get("ps_properties", None)
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
            print(f"Splitting a pRef of size {pRef_to_split.sample_size}")
            if (current_depth >= self.maximum_depth) or (pRef_to_split.sample_size < 2):
                return PSRegressionTreeLeafNode.from_pRef(pRef_to_split)

            # otherwise we split more
            node = PSRegressionTreeBranchNode.from_pRef(pRef_to_split)
            splitting_ps = PSRegressionTreeBranchNode.find_splitting_ps(search_settings=self.search_settings,
                                                                        pRef=pRef_to_split,
                                                                        ancestors=ancestors)

            if self.search_settings.verbose:
                print(f"The splitting PS is {splitting_ps}")
            node.split_ps = splitting_ps
            matching_indexes = pRef_to_split.get_indexes_matching_ps(splitting_ps)
            matching_pRef, not_matching_pRef = pRef_to_split.split_by_indexes(matching_indexes)

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
            node.ps_properties = descriptors

    def all_pss_as_list(self) -> list[PS]:
        return [node.split_ps for node in self.all_nodes_as_list() if isinstance(node, PSRegressionTreeBranchNode)]

    @classmethod
    def default_properties_repr(cls, properties: list[(str, float, float)]) -> str:
        return "\n".join(
            f"{prop_name} = {prop_value:.2f}-> {prop_rank:.2f}" for prop_name, prop_value, prop_rank in
            properties)


    def get_custom_reprs(self) -> (Callable, Callable):
        if self.problem is None:
            repr_ps = repr
            repr_properties = PSRegressionTree.default_properties_repr
        else:
            repr_ps = self.problem.repr_ps
            if hasattr(self.problem, "repr_descriptors"):
                repr_properties = self.problem.repr_descriptors
            else:
                repr_properties = PSRegressionTree.default_properties_repr

        return repr_ps, repr_properties

    def __repr__(self):
        repr_ps, repr_properties = self.get_custom_reprs()

        if self.root_node is None:
            return "Invalid Tree"

        return self.root_node.repr_custom(repr_ps, repr_properties)

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

    @classmethod
    def from_file(cls, filename: str):
        with open(filename, "r") as file:
            data = json.load(file)
        return cls.from_dict(data)

    def to_file(self, filename: str):
        with utils.open_and_make_directories(filename) as file:
            data = self.as_dict()
            json.dump(data, file, indent=4)

    def with_permutation(self, permutation: list[int]):
        def permute_ps(ps: PS) -> PS:
            return PS(ps.values[permutation])

        def permute_node(node: PSRegressionTreeNode) -> PSRegressionTreeNode:
            if isinstance(node, PSRegressionTreeBranchNode):
                result = PSRegressionTreeBranchNode(prediction=node.prediction,
                                                    other_statistics=node.other_statistics)
                result.split_ps = permute_ps(node.split_ps)
                result.matching_branch = permute_node(node.matching_branch)
                result.not_matching_branch = permute_node(node.not_matching_branch)
                result.ps_properties = None  # because their names might have changed...
                return result
            else:
                return node

        result = PSRegressionTree(maximum_depth=self.maximum_depth)
        result.root_node = permute_node(self.root_node)
        result.search_settings = self.search_settings
        result.problem = None  # this needs to be set somewhere else, since the problem will be different
        return result


    def print_ASCII(self, show_not_matching_nodes: bool = True):
        repr_ps, repr_properties = self.get_custom_reprs()
        def add_node_repr(node: PSRegressionTreeNode, parent, preamble: str):
            own_node_repr = Node(preamble+"\n"+node.get_node_text(repr_ps, repr_properties), parent = parent)
            if isinstance(node, PSRegressionTreeBranchNode):
                add_node_repr(node.matching_branch, parent = own_node_repr, preamble = "Matching")
                if show_not_matching_nodes:
                    add_node_repr(node.not_matching_branch, parent = own_node_repr, preamble = "NOT matching")
            return own_node_repr

        root_node_repr = add_node_repr(self.root_node, parent = None, preamble="Root")
        for pre, fill, node in RenderTree(root_node_repr):
            lines = node.name.splitlines()
            # Print the first line with the usual prefix.
            print(f"{pre}{lines[0]}")
            # For any additional lines, print them with an indentation that matches the node's position.
            for line in lines[1:]:
                print(f"{fill}{line}")
