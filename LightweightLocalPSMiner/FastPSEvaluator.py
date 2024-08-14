from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.PSMetric.Metric import Metric
from Core.PSMetric.Simplicity import Simplicity
from LightweightLocalPSMiner.SolutionRowCacher import CachedRowsNode, fast_get_mean_fitness
from LinkageExperiments.LocalLinkage import BivariateVariance


class FastPSEvaluator:
    pRef: PRef
    row_cacher_root_node: CachedRowsNode
    atomicity_metric: Metric
    simplicity_metric: Simplicity

    used_evaluations: int


    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.row_cacher_root_node = CachedRowsNode.root_node_from_pRef(pRef)
        self.atomicity_metric = BivariateVariance()
        self.atomicity_metric.set_pRef(pRef)
        self.simplicity_metric = Simplicity()

        self.used_evaluations = 0



    def get_S_MF_A(self, ps: PS) -> (float, float, float):
        simplicity = self.simplicity_metric.get_single_score(ps)
        mean_fitness = fast_get_mean_fitness(ps, self.row_cacher_root_node)
        atomicity = self.atomicity_metric.get_single_score(ps)

        self.used_evaluations +=1

        return (simplicity, mean_fitness, atomicity)