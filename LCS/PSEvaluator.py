from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.LocalPerturbation import PerturbationOfSolution
from Core.PSMetric.Metric import Metric


class GeneralPSEvaluator:
    fitness_p_value_metric: MannWhitneyU

    local_linkage_metric: PerturbationOfSolution
    used_evaluations: int

    mean_fitness_metric: Metric

    def __init__(self,
                 pRef: PRef):
        self.used_evaluations = 0

        self.fitness_p_value_metric = MannWhitneyU()
        self.fitness_p_value_metric.set_pRef(pRef)

        self.mean_fitness_metric = MeanFitness()
        self.mean_fitness_metric.set_pRef(pRef)

        self.local_linkage_metric = PerturbationOfSolution()
        self.local_linkage_metric.set_pRef(pRef)

    def set_solution(self, solution: FullSolution):
        #self.linkage_metric.set_solution(solution)
        self.local_linkage_metric.set_solution(solution)
