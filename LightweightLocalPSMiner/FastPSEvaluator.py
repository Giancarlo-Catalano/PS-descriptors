from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Additivity import MutualInformation
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.PSMetric.Metric import Metric
from Core.PSMetric.Simplicity import Simplicity
from Core.PSMetric.ValueSpecificMutualInformation import ValueSpecificMutualInformation, \
    SolutionSpecificMutualInformation
from LightweightLocalPSMiner.SolutionRowCacher import CachedRowsNode, fast_get_mean_fitness
from LinkageExperiments.LocalLinkage import BivariateVariance


class FastPSEvaluator:
    pRef: PRef
    row_cacher_root_node: CachedRowsNode
    atomicity_metric: Metric
    simplicity_metric: Simplicity

    sign_of_fitness: float
    used_evaluations: int

    def __init__(self,
                 pRef: PRef,
                 sign_of_fitness: float = 1):
        self.pRef = pRef
        self.sign_of_fitness = sign_of_fitness
        self.row_cacher_root_node = CachedRowsNode.root_node_from_pRef(pRef)
        self.atomicity_metric = SolutionSpecificMutualInformation()
        self.atomicity_metric.set_pRef(pRef)
        self.simplicity_metric = Simplicity()

        self.used_evaluations = 0

    def set_to_minimise_fitness(self):
        self.sign_of_fitness = -1

    def set_to_maximise_fitness(self):
        self.sign_of_fitness = 1

    def get_mean_fitness(self, ps: PS) -> float:
        return self.sign_of_fitness * fast_get_mean_fitness(ps, self.row_cacher_root_node)

    def get_S_MF_A(self, ps: PS) -> (float, float, float):
        simplicity = self.simplicity_metric.get_single_score(ps)
        mean_fitness = self.get_mean_fitness(ps)
        atomicity = self.atomicity_metric.get_single_score(ps)

        self.used_evaluations += 1

        return simplicity, mean_fitness, atomicity
