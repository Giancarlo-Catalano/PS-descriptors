import random

import numpy as np
from deap import creator
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace
from PSMiners.Mining import get_history_pRef
from PSMiners.PyMoo.Operators import PSPolynomialMutation, PSGeometricSampling, PSSimulatedBinaryCrossover


class PSPyMooProblem(ElementwiseProblem):
    pRef: PRef
    objectives_evaluator: Classic3PSEvaluator


    def __init__(self,
                 pRef: PRef):
        self.pRef = pRef
        self.objectives_evaluator = Classic3PSEvaluator(self.pRef)

        lower_bounds = np.full(shape=self.search_space.amount_of_parameters, fill_value=-1)  # the stars
        upper_bounds = self.search_space.cardinalities - 1
        super().__init__(n_var = self.search_space.amount_of_parameters,
                         n_obj=3,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=int)

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    def individual_to_ps(self, x):
        return PS(x)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -self.objectives_evaluator.get_S_MF_A(self.individual_to_ps(x))  # minus sign because it's a maximisation task




def pymoo_result_to_pss(res) -> list[PS]:
    return [PS(row) for row in res.X]



def apply_to_nsgaiii_pymoo(benchmark_problem: BenchmarkProblem, pRef: PRef):
    pymoo_problem = PSPyMooProblem(pRef)

    print("Running the NSGA using pymoo")
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    # create the algorithm object
    algorithm = NSGA3(pop_size=600,
                      ref_dirs=ref_dirs,
                      sampling=PSGeometricSampling(),
                      crossover=PSSimulatedBinaryCrossover(),
                      mutation=PSPolynomialMutation(benchmark_problem.search_space),
                      eliminate_duplicates=True,
                      )

    # execute the optimization
    res = minimize(pymoo_problem,
                   algorithm,
                   seed=1,
                   termination=('n_gen', 100))

    Scatter().add(res.F).show()

    pss = pymoo_result_to_pss(res)
    print(f"The pss found are {len(pss)}:")
    for ps in pss:
        print(ps)


    print("Function value: %s" % res.F)


def test_pymoo(benchmark_problem: BenchmarkProblem):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            sample_size=1000,
                            which_algorithm="SA",
                            verbose=True)

    apply_to_nsgaiii_pymoo(benchmark_problem, pRef)


