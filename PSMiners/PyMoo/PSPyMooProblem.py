import random
from typing import Any

import numpy as np
from deap import creator
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
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
from PSMiners.PyMoo.CustomCrowding import PyMooCustomCrowding, PyMooPSGenotypeCrowding, PyMooPSSequentialCrowding
from PSMiners.PyMoo.FitnessSharing import get_sharing_scores
from PSMiners.PyMoo.Operators import PSPolynomialMutation, PSGeometricSampling, PSSimulatedBinaryCrossover
from utils import announce


class PSPyMooProblem(Problem):
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

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.objectives_evaluator.get_S_MF_A(self.individual_to_ps(row)) for row in X])
        out["F"] = -metrics  # minus sign because it's a maximisation task

        # sharing_values = get_sharing_scores(X, 0.5, 12)
        # out["F"] /= (sharing_values.reshape((-1, 1)))+1




def pymoo_result_to_pss(res) -> list[PS]:
    return [PS(row) for row in res.X]



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
                     survival=survival,
                     selection=TournamentSelection(pressure=pop_size // 10)
                      )
    if which_algorithm == "NSGAIII":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=20)

        # create the algorithm object
        return NSGA3(pop_size=pop_size,
                          ref_dirs=ref_dirs,
                          sampling=PSGeometricSampling(),
                          crossover=PSSimulatedBinaryCrossover(),
                          mutation=PSPolynomialMutation(ss),
                          eliminate_duplicates=True,
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

def test_pymoo(benchmark_problem: BenchmarkProblem, pRef: PRef, which_algorithm: str, which_crowding: str, ngen=100):


    algorithm = get_pymoo_algorithm(pRef, which_algorithm = which_algorithm, which_crowding = which_crowding)
    pymoo_problem = PSPyMooProblem(pRef)

    with announce(f"Running {which_algorithm} using {which_crowding}"):
        res = minimize(pymoo_problem,
                       algorithm,
                       seed=1,
                       termination=('n_gen', ngen),
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


def sequential_search_pymoo(pRef: PRef,
                            ps_budget_per_run: int,
                            pop_size: int = 150,
                            runs: int = 6):
    initial_crowding_operator = RankAndCrowding(crowding_func = "ce")

    current_crowding_operator = initial_crowding_operator
    accumulated_winners = []
    for iteration in range(runs):
        algorithm =  NSGA2(pop_size=pop_size,
                     sampling=PSGeometricSampling(),
                     crossover=PSSimulatedBinaryCrossover(),
                     mutation=PSPolynomialMutation(pRef.search_space),
                     eliminate_duplicates=True,
                     survival=current_crowding_operator
                     )
        pymoo_problem = PSPyMooProblem(pRef)

        with announce(f"Running {NSGA2} for iteration #{iteration}"):
            res = minimize(pymoo_problem,
                           algorithm,
                           seed=1,
                           termination=('n_evals', ps_budget_per_run))


        e_pss = [EvaluatedPS(values, metric_scores=ms) for values, ms in zip(res.X, res.F)]
        e_pss.sort(reverse=False, key=lambda x: x.metric_scores[-1])  # sorting by atomicity, resverse=False because it's a minimisation task
        winners = e_pss[:3]
        print(f"The winners for the {iteration}th run are")
        for winner in winners:
            print(winner)

        accumulated_winners.extend(winners)
        current_crowding_operator = PyMooPSSequentialCrowding(accumulated_winners)
        if all(x==1 for x in current_crowding_operator.coverage):
            print(f"Search terminated because the entire space has been covered")
            break

    accumulated_winners.sort(reverse=False, key=lambda x: x.metric_scores[-1]) # false because it's a minimisation task

    print(f"At the end, the winners are")
    for winner in accumulated_winners:
        print(winner)

