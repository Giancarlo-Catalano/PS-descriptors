from typing import Iterable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.MeanFitness import FitnessDelta
from Core.PSMetric.SignificantlyHighAverage import SignificantlyHighAverage, MannWhitneyU
from Core.PSMetric.ValueSpecificMutualInformation import SolutionSpecificMutualInformation, \
    FasterSolutionSpecificMutualInformation, NotValueSpecificMI
from LightweightLocalPSMiner.Operators import LocalPSGeometricSampling, ObjectiveSpaceAvoidance


class TMEvaluator:

    linkage_metric: SolutionSpecificMutualInformation
    delta_fitness_metric: FitnessDelta
    fitness_p_value_metric: MannWhitneyU
    global_mean_fitness: float
    used_evaluations: int

    def __init__(self,
                 pRef: PRef,
                 use_value_specific_linkage: bool = False):
        self.used_evaluations = 0
        if use_value_specific_linkage:
            self.linkage_metric = FasterSolutionSpecificMutualInformation()
        else:
            self.linkage_metric = NotValueSpecificMI()
        self.linkage_metric.set_pRef(pRef)

        self.delta_fitness_metric = FitnessDelta()
        self.delta_fitness_metric.set_pRef(pRef)

        self.fitness_p_value_metric = MannWhitneyU()
        self.fitness_p_value_metric.set_pRef(pRef)

        self.global_mean_fitness = np.average(pRef.fitness_array)


    def get_A_D(self, ps: PS) -> (float, float):
        self.used_evaluations += 1
        atomicity = self.linkage_metric.get_atomicity_score(ps)
        dependence = self.linkage_metric.get_dependence_score(ps)

        return -atomicity, dependence


    def set_solution(self, solution: FullSolution):
        self.linkage_metric.set_solution(solution)


    def is_ps_beneficial(self, ps: PS) -> bool:
        self.used_evaluations += 1
        return self.delta_fitness_metric.get_single_score(ps) > 0

class TMLocalPymooProblem(Problem):

    tm_evaluator: TMEvaluator

    def __init__(self,
                 solution_to_explain: FullSolution,
                 objectives_evaluator: TMEvaluator):
        self.solution_to_explain = solution_to_explain
        self.objectives_evaluator = objectives_evaluator
        self.objectives_evaluator.set_solution(solution_to_explain)

        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds+1
        super().__init__(n_var = n_var,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.objectives_evaluator.get_A_D(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics


def local_tm_ps_search(to_explain: FullSolution,
                    to_avoid: Iterable[PS],
                    ps_evaluator: TMEvaluator,
                    ps_budget: int,
                    population_size: int,
                    verbose=True) -> list[PS]:
    problem = TMLocalPymooProblem(to_explain,
                                  ps_evaluator)


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

    # then we arrange them in such a way that if a really good one is found, it will be the first one

    correct_sign_pss = []
    wrong_signs = []
    for ps in result_pss:
        a, d  = ps.metric_scores
        if a < 0 and d > 0:
            correct_sign_pss.append(ps)
        else:
            wrong_signs.append(ps)

    def get_atomicity_minus_dependence(ps: EvaluatedPS) -> float:
        a, d = ps.metric_scores
        return -d + a
    #correct_sign_pss.sort(key=get_atomicity_minus_dependence, reverse=True)
    #wrong_signs.sort(key=get_atomicity_minus_dependence, reverse=True)

    # budget_after_run = ps_evaluator.used_evaluations
    # print(f"The budget used for this run was {budget_after_run-budget_before_run}")
    # print(f"The pss with the correct sign are {len(correct_sign_pss)}")
    # for ps in correct_sign_pss:
    #     print("\t", ps)
    # print(f"The pss with the wrong sign are {len(wrong_signs)}")
    # for ps in wrong_signs:
    #     print("\t", ps)

    if verbose:
        print(f"The search for PSs in {to_explain} resulted in ")
        for ps in result_pss:
            print("\t", ps)

    if len(correct_sign_pss) > 0:
        return correct_sign_pss
    else:
        return wrong_signs

def test_local_search():
    problem = Trapk(4, 4)

    already_mined = [] #[PS([1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
                     #PS([-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])]

    to_explain = FullSolution([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0])

    pRef = problem.get_reference_population(10000)

    tm_evaluator = TMEvaluator(pRef)

    results = local_tm_ps_search(to_explain = to_explain,
                              to_avoid=already_mined,
                              ps_evaluator = tm_evaluator,
                              ps_budget=1000,
                              population_size=50,
                              verbose=True)

    print("results of search:")
    for result in results:
        print(result)

    print("The results by dependence-atomicity are")
    results.sort(key=lambda x: x.metric_scores[0]-x.metric_scores[1], reverse=True)
    for result in results:
        a, d = result.metric_scores
        if a < 0 and d > 0:
            print(result)

