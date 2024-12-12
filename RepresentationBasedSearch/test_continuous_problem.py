import numpy as np

from RepresentationBasedSearch.ContinuousVariables.ContinuousVariableProblem import ContinuousOptimisationProblem, \
    ContinuousSearchSpace, apply_ga_to_find_maximum


class JCAssignmentProblem(ContinuousOptimisationProblem):
    def __init__(self):
        super().__init__(search_space=ContinuousSearchSpace(lower_bounds=[-1 for _ in range(3)],
                                                            upper_bounds=[1 for _ in range(3)]))

    def __repr__(self):
        return "JCAssignment"

    def fitness_function(self, values: np.ndarray):
        x, y, z = values
        return 2 * np.sin(np.cos(5 * x + 5 * y)) + z * np.sin(np.cos(11 * z)) - 0.5 * x * np.cos(7 * z) - (
                    x ** 2 + y ** 2 + z) / 10


def test_solve_problem():
    problem = JCAssignmentProblem()
    result, fitness = apply_ga_to_find_maximum(problem, tournament_size=10, budget=10000, return_pRef=False)

    x, y, z = result

    print("The best result is")
    print(f"{x = }")
    print(f"{y = }")
    print(f"{z = }")

    print(f"{fitness = }")


test_solve_problem()