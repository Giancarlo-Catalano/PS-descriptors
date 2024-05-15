import functools
from typing import Optional, Iterable

from Core.PS import PS


@functools.total_ordering
class EvaluatedPS(PS):
    ps: PS
    metric_scores: Optional[list[float]]
    aggregated_score: Optional[float]

    def __init__(self, values: Iterable[int], metric_scores=None, aggregated_score=None):
        super().__init__(values)
        self.metric_scores = metric_scores
        self.aggregated_score = aggregated_score

    def __repr__(self):
        result = f"{PS(self.values)}"
        if self.metric_scores is not None:
            result += "["
            for metric in self.metric_scores:
                result += f"{metric:.3f}, "
            result += "]"
        if self.aggregated_score is not None:
            result += f", aggregated_score = {self.aggregated_score:.3f}"
        return result

    def __lt__(self, other):
        return self.aggregated_score < other.aggregated_score
