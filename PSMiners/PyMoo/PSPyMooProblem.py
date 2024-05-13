import random

import numpy as np
from deap import creator
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace
from PSMiners.Mining import get_history_pRef
from PSMiners.PyMoo.CustomCrowding import PyMooCustomCrowding, PyMooPSGenotypeCrowding
from PSMiners.PyMoo.Operators import PSPolynomialMutation, PSGeometricSampling, PSSimulatedBinaryCrossover
from utils import announce


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


def gc_crowding(F, filter_out_duplicates=None, n_remove=None, **kwargs):
    print("Called gc_crowding")


    pop = kwargs["pop"]
    front_indexes = kwargs["front_indexes"]
    pop_matrix = np.array([ind.X for ind in pop])
    where_fixed = pop_matrix != STAR
    counts = np.sum(where_fixed, axis=0)
    foods = (1 / counts).reshape((-1, 1))
    scores = where_fixed[front_indexes] @ foods
    result = scores.ravel()
    colour = "green"

    return result

def get_pymoo_algorithm(pRef,
                        which_algorithm: str,
                        pop_size: int = 100,
                        which_crowding: str= "cd"):
    ss = pRef.search_space
    n = ss.amount_of_parameters

    if which_crowding == "gc":
        survival = PyMooPSGenotypeCrowding()
    else:
        survival = RankAndCrowding(crowding_func = which_crowding)
    if which_algorithm == "NSGAII":
        return NSGA2(pop_size=pop_size,
                      sampling=PSGeometricSampling(),
                      crossover=PSSimulatedBinaryCrossover(),
                      mutation=PSPolynomialMutation(ss),
                      eliminate_duplicates=True,
                     survival=survival
                      )
    if which_algorithm == "NSGAIII":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

        # create the algorithm object
        return NSGA3(pop_size=pop_size,
                          ref_dirs=ref_dirs,
                          sampling=PSGeometricSampling(),
                          crossover=PSSimulatedBinaryCrossover(),
                          mutation=PSPolynomialMutation(ss),
                          eliminate_duplicates=True,
                     survival=survival,
                          )
    elif which_algorithm == "MOEAD":
        ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

        return MOEAD(
            ref_dirs = ref_dirs,
            sampling=PSGeometricSampling(),
            crossover=PSSimulatedBinaryCrossover(),
            mutation=PSPolynomialMutation(ss),
            n_neighbors=pRef.search_space.amount_of_parameters,
            prob_neighbor_mating=0.7,
            survival=survival
        )

def test_pymoo(benchmark_problem: BenchmarkProblem, pRef: PRef, which_algorithm: str, which_crowding: str):


    algorithm = get_pymoo_algorithm(pRef, which_algorithm = which_algorithm, which_crowding = which_crowding)
    pymoo_problem = PSPyMooProblem(pRef)

    with announce(f"Running {which_algorithm} using {which_crowding}"):
        res = minimize(pymoo_problem,
                       algorithm,
                       seed=1,
                       termination=('n_gen', 200),
                       verbose=True)


    pss = pymoo_result_to_pss(res)
    e_pss = [EvaluatedPS(values, metric_scores=ms) for values, ms in zip(res.X, res.F)]
    e_pss.sort(reverse=False, key=lambda x: x.metric_scores[-1])  # sorting by atomicity, resverse=False because it's a minimisation task
    #Scatter().add(res.F).show()

    pss = pymoo_result_to_pss(res)
    print(f"The pss found are {len(pss)}:")
    for e_ps in e_pss:
        print(e_ps)


    return pss


