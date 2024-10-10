from typing import Optional

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.Additivity import MutualInformation
from Core.PSMetric.Linkage.LocalPerturbation import PerturbationOfSolution
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.PSMetric.Metric import Metric


class GeneralPSEvaluator:
    fitness_p_value_metric: MannWhitneyU

    used_evaluations: int

    mean_fitness_metric: Metric

    traditional_linkage: TraditionalPerturbationLinkage

    def __init__(self,
                 pRef: Optional[PRef],
                 optimisation_problem: BenchmarkProblem):
        self.used_evaluations = 0

        self.fitness_p_value_metric = MannWhitneyU()

        self.mean_fitness_metric = MeanFitness()

        self.traditional_linkage = TraditionalPerturbationLinkage(optimisation_problem)


        if pRef is not None:
            self.set_pRef(pRef)


    def set_pRef(self, pRef: PRef):
        for metric in [self.fitness_p_value_metric, self.mean_fitness_metric]:
            metric.set_pRef(pRef)
    def set_solution(self, solution: FullSolution):
        self.traditional_linkage.set_solution(solution)


