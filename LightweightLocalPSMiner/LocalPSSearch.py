from typing import Iterable

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PS import PS
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearchProblem import LocalPSPymooProblem
from LightweightLocalPSMiner.Operators import LocalPSGeometricSampling, ObjectiveSpaceAvoidance
from PSMiners.AbstractPSMiner import AbstractPSMiner

def local_ps_search(to_explain: FullSolution,
                    to_avoid: Iterable[PS],
                    ps_evaluator: FastPSEvaluator,
                    ps_budget: int,
                    population_size: int,
                    verbose=True) -> list[PS]:
    problem = LocalPSPymooProblem(to_explain,
                                  ps_evaluator)

    # before_run_budget = ps_evaluator.used_evaluations

    algorithm = NSGA2(pop_size=population_size,
                      sampling=LocalPSGeometricSampling(),
                      crossover=SimulatedBinaryCrossover(prob=0.5),
                      mutation=BitflipMutation(prob=1/problem.n_var),
                      eliminate_duplicates=True,
                      survival=ObjectiveSpaceAvoidance(to_avoid)
                      )

    res = minimize(problem,
                   algorithm,
                   termination=('n_evals', ps_budget),
                   verbose=verbose)

    result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=ms)
                  for values, ms in zip(res.X, res.F)]

    return result_pss



def test_lightweight_miner():
    problem = RoyalRoad(4, 4)

    already_mined = [PS([1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                     PS([-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])]

    to_explain = FullSolution([1 for i in range(16)])

    pRef = problem.get_reference_population(10000)

    ps_evaluator = FastPSEvaluator(pRef)

    results = local_ps_search(to_explain = to_explain,
                              to_avoid=already_mined,
                              ps_evaluator = ps_evaluator,
                              ps_budget=3000,
                              population_size=50,
                              verbose=True)

    print("results of search:")
    results.sort(key=lambda x: x.metric_scores[2])
    for result in results:
        print(result)


