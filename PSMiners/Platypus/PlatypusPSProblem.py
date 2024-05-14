from platypus import Problem, Integer, NSGAII, NSGAIII, CMAES, GDE3, IBEA, MOEAD, OMOPSO, SMPSO, SPEA2, EpsMOEA

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace
from utils import announce


class PlatypusPSProblem(Problem):
    pRef: PRef
    evaluator: Classic3PSEvaluator
    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.evaluator = Classic3PSEvaluator(self.pRef)
        super().__init__(nvars = self.search_space.amount_of_parameters,
                         nobjs=3)
        for index, cardinality in enumerate(self.search_space.cardinalities):
            self.types[index] = Integer(-1, cardinality-1)

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space


    def evaluate(self, solution):
        ps = PS(solution.variables[:])
        solution.objectives[:] = list(-self.evaluator.get_S_MF_A(ps))


    def solution_to_ps(self, solution) -> PS:
        values = [v_type.decode(solution_v)
                  for v_type, solution_v in zip(self.types, solution.variables)]
        return PS(values)



def make_platypus_algorithm(which_algorithm, problem):
    match which_algorithm:
        case "NSGAII": return NSGAII(problem)
        case "NSGAIII": return NSGAIII(problem, divisions_outer=12)
        case "CMAES": return CMAES(problem, epsilons=[0.05])
        case "GDE3": return GDE3(problem)
        case "IBEA": return IBEA(problem)
        case "MOEAD": return MOEAD(problem)
        case "OMOPSO": return OMOPSO(problem, epsilons=[0.05])
        case "SMPSO": return SMPSO(problem)
        case "SPEA2": return SPEA2(problem)
        case "EpsMOEA": return EpsMOEA(problem, epsilons=[0.05])

def test_platypus(benchmark_problem: BenchmarkProblem,
                  pRef: PRef,
                  budget: int,
                  which_algorithm: str):
    problem = PlatypusPSProblem(pRef)

    algorithm = make_platypus_algorithm(which_algorithm, problem)


    algorithm.run(budget)

    colour = "green"
    return list(map(problem.solution_to_ps, algorithm.result))