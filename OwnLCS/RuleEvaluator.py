from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Additivity import MutualInformation
from Core.PSMetric.CleanLinkage import CleanLinkage
from Core.PSMetric.MeanFitness import MeanFitness
from Core.PSMetric.Metric import Metric
from Core.PSMetric.Simplicity import Simplicity
from OwnLCS.Rule import Rule
from LinkageExperiments.LocalLinkage import LocalLinkage, BivariateVariance


class RuleEvaluator:
    simplicity_metric: Simplicity
    mean_fitness_metric: MeanFitness  # note that this might be replaced by accuracy in future implementatons
    linkage_metric: Metric

    def __init__(self,
                 simplicity_metric: Simplicity,
                 mean_fitness_metric: MeanFitness,
                 linkage_metric: Metric):
        self.simplicity_metric = simplicity_metric
        self.mean_fitness_metric = mean_fitness_metric
        self.linkage_metric = linkage_metric

    @classmethod
    def from_pRef(cls, pRef: PRef):
        simplicity = Simplicity()
        mean_fitness = MeanFitness()
        atomicity = BivariateVariance()

        for metric in [simplicity, mean_fitness, atomicity]:
            metric.set_pRef(pRef)

        return cls(simplicity, mean_fitness, atomicity)

    @classmethod
    def get_adjusted_accuracy(cls, rule: Rule) -> float:
        if rule.amount_of_matches == 0:
            return 0
        else:
            return (1 - 0.5 / rule.amount_of_matches) * (
                    rule.amount_of_correct / rule.amount_of_matches)
    def get_metrics(self, rule: Rule) -> (float, float, float):

        return (1,#rule.cached_simplicity,
                self.get_adjusted_accuracy(rule),
                1) # rule.cached_linkage)
