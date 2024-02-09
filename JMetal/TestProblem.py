import copy
import random
from abc import ABC

import numpy as np
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.operator import Crossover
from jmetal.core.problem import Problem, BinaryProblem, IntegerProblem
from jmetal.core.solution import BinarySolution, IntegerSolution
from jmetal.operator import PolynomialMutation, SBXCrossover, SPXCrossover, BitFlipMutation, IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from SearchSpace import SearchSpace


class SubsetSum(BinaryProblem):
    C: int
    W: list[int]
    number_of_bits: int

    def __init__(self, C: int, W: list[int]):
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 1
        self.number_of_variables = 1  # why?

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Sum"]

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return 1

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        def maximisation_score(obtained_sum) -> float:
            if obtained_sum > self.C:
                return self.C - obtained_sum * 0.1
            if obtained_sum < 0:
                return 0
            return obtained_sum

        total_sum = sum([item for item, is_used in zip(self.W, solution.variables[0]) if is_used])

        # set the objectives in the solution, and return it
        solution.objectives[0] = -1.0 * maximisation_score(total_sum)
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)

        new_solution.variables[0] = [bool(random.getrandbits(1)) for _ in range(self.number_of_bits)]
        return new_solution

    def name(self) -> str:
        return "Subset Sum"


class GianIntegerProblem(IntegerProblem, ABC):
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.lower_bound = [0 for _ in self.search_space.cardinalities]
        self.upper_bound = [cardinality - 1 for cardinality in self.search_space.cardinalities]

        super(GianIntegerProblem, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Sum", "Product"]

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return 2

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        score_sum = sum(solution.variables[0])
        score_product = np.product(solution.variables[0])
        solution.objectives[0] = score_sum
        solution.objectives[1] = score_product
        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives(),
            self.number_of_constraints())
        new_solution.variables[0] = [random.randrange(lower, upper) for lower, upper in
                                     zip(self.lower_bound, self.upper_bound)]

        return new_solution

    def name(self) -> str:
        return "Subset Sum"


class BoringIntegerProblem(IntegerProblem, ABC):
    lower_bound = [2, 3, 4]
    upper_bound = [8, 9, 10]

    def __init__(self):
        super(BoringIntegerProblem, self).__init__()

        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Sum", "Product"]
        self.lower_bound = [2, 3, 4]
        self.upper_bound = [8, 9, 10]

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return 2

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        score_sum = sum(solution.variables)
        score_product = np.product(solution.variables)
        solution.objectives[0] = score_sum
        solution.objectives[1] = score_product
        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [random.randrange(lower, upper)
                                     for lower, upper in zip(self.lower_bound, self.upper_bound)]

        return new_solution

    def name(self) -> str:
        return "Boring Integer problem"


def test_JMetal_1():
    problem = SubsetSum(21, [1, 2, 4, 8, 16, 32, 64])

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(probability=1 / problem.number_of_bits),
        crossover=IntegerSBXCrossover(probability=1, distribution_index = 20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000))

    print("Setup the algorithm, now we run it")

    algorithm.run()

    print("The algorithm has stopped, the results are")

    front = get_non_dominated_solutions(algorithm.get_result())

    for item in front:
        value = sum([value for value, is_used in zip([1, 2, 4, 8, 16, 32, 64], item.variables[0]) if is_used])
        print(f"{item}, with value {value}\n")


def test_JMetal_integer():
    problem = BoringIntegerProblem()
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1 / 3, distribution_index=20),
        crossover=IntegerSBXCrossover(probability=1, distribution_index = 20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000))

    print("Setup the algorithm, now we run it")

    algorithm.run()

    print("The algorithm has stopped, the results are")

    front = get_non_dominated_solutions(algorithm.get_result())

    for item in front:
        print(f"{item}\n")
