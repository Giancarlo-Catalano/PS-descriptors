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

    local_linkage_metric: PerturbationOfSolution
    used_evaluations: int

    mean_fitness_metric: Metric

    traditional_linkage: TraditionalPerturbationLinkage

    def __init__(self,
                 pRef: PRef,
                 optimisation_problem: BenchmarkProblem):
        self.used_evaluations = 0

        self.fitness_p_value_metric = MannWhitneyU()
        self.fitness_p_value_metric.set_pRef(pRef)

        self.mean_fitness_metric = MeanFitness()
        self.mean_fitness_metric.set_pRef(pRef)

        # self.local_linkage_metric = PerturbationOfSolution()
        # self.local_linkage_metric.set_pRef(pRef)

        self.traditional_linkage = TraditionalPerturbationLinkage(optimisation_problem)

    def set_solution(self, solution: FullSolution):
        #self.local_linkage_metric.set_solution(solution)
        self.traditional_linkage.set_solution(solution)


    def set_positive_or_negative(self, search_for_negative_traits: bool):
        self.fitness_p_value_metric.search_for_negative_traits = search_for_negative_traits
