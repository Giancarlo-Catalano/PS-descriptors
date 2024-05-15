import random
from typing import Union, Sequence, Any

import numpy as np

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace
from PSMiners.DEAP.deap_utils import geometric_distribution_values_of_ps
from PSMiners.MOEAD.Library.core.genetic_operator.abstract_operator import GeneticOperator
from PSMiners.MOEAD.Library.core.genetic_operator.numerical.polynomial_mutation import PolynomialMutation
from PSMiners.MOEAD.Library.core.offspring_generator.abstract_mating import OffspringGenerator
from PSMiners.MOEAD.Library.problem.problem import Problem
from PSMiners.MOEAD.Library.solution.base import Solution
from PSMiners.MOEAD.Library.solution.one_dimension_solution import OneDimensionSolution


class PSPolynomialMutation(PolynomialMutation):
    """
    Polynomial Mutation operator.

    Require only one solution.
    """


    search_space: SearchSpace
    params: Any


    def __init__(self, search_space: SearchSpace, **params):
        print(f"Constructing the PSPolynomialMutation({SearchSpace}) operator")
        self.search_space = search_space
        super().__init__(**params)


    def set_params(self, params):
        self.params = params

    def run(self):
        """
        Execute the genetic operator

        :return: {:class:`~moead_framework.solution.one_dimension_solution.OneDimensionSolution`} the offspring
        """

        self.number_of_solution_is_correct(n=2)
        solution = self.solutions[0]  # currently it ignores the second one...

        return self.mutation(s=solution, rate=1 /len(solution))

    def mutation(self, s: list[int], rate) -> list[int]:
        new_values = s.copy()
        for index, cardinality in enumerate(self.search_space.cardinalities):
            if random.random() < rate:
                new_values[index] = random.randrange(cardinality+1)-1
        return new_values



class MOEADPSProblem(Problem):
    pRef: PRef
    ps_evaluator: Classic3PSEvaluator



    def __init__(self, pRef: PRef):
        super().__init__(objective_number=3)
        self.pRef = pRef
        self.ps_evaluator = Classic3PSEvaluator(self.pRef)
        self.dtype = int


    @classmethod
    def decision_vector_to_PS(cls, decision_vector: np.ndarray) -> PS:
        return PS(decision_vector)


    def f(self, function_id: int, decision_vector: np.ndarray):
        ps = MOEADPSProblem.decision_vector_to_PS(decision_vector)
        print(f"Calling f({function_id}, {ps})")

        return function_id


    def evaluate(self, x: Union[Solution, Sequence]) -> OneDimensionSolution:
        """
        Evaluate the given solution for the current problem and store the outcome

        :param x: A {Solution} containing all decision variables
        :return: :class:`~moead_framework.solution.one_dimension_solution.OneDimensionSolution`
        """
        if not isinstance(x, OneDimensionSolution):
            x = OneDimensionSolution(np.array(x, dtype=self.dtype))

        ps = MOEADPSProblem.decision_vector_to_PS(x.decision_vector)
        metrics = -self.ps_evaluator.get_S_MF_A(ps)  # we have to negate them because
        x.F = list(metrics)
        return x


    def generate_random_solution(self):
        values = geometric_distribution_values_of_ps(self.pRef.search_space)
        return self.evaluate(list(values))


