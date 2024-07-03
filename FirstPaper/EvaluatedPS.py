import functools
from typing import Optional, Iterable

import numpy as np

from FirstPaper.PS import PS


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


def write_evaluated_pss_to_file(e_pss: list[EvaluatedPS], file: str):
    """
    Similar to the function above, but we also write the objective values
    @param e_pss: The evaluated partial Solutions
    @param file: the file to store the pss in
    @return: Nothing!
    """
    ps_matrix = np.array([e_ps.values for e_ps in e_pss])
    fitness_matrix = np.array([e_ps.metric_scores for e_ps in e_pss])

    np.savez(file, ps_matrix = ps_matrix, fitness_matrix=fitness_matrix)


def load_pss(file: str) -> list[[EvaluatedPS | PS]]:
    """
    Loads the partial solutions, and checks automatically which kind is stored (whether they are evaluated or not)
    @param file: the file to read from
    @return: the list of partial solutions, evaluated or not
    """
    results_dict = np.load(file)
    ps_matrix = results_dict["ps_matrix"]

    pss = [PS(row) for row in ps_matrix]

    if "fitness_matrix" in results_dict:
        fitness_matrix = results_dict["fitness_matrix"]
        return[EvaluatedPS(ps, metric_scores=list(fitness_values))
                 for ps, fitness_values in zip(pss, fitness_matrix)]
    else:
        return pss
