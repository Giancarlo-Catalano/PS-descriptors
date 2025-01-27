from typing import Optional, Callable

import numpy as np

import utils
from Core.PS import PS, STAR
from Core.PSMetric.Linkage import LocalPerturbation
from Core.PSMetric.Linkage.LocalPerturbation import PerturbationOfSolution
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from LCS.PSEvaluator import GeneralPSEvaluator


def filter_pss(pss: list[PS],
               ps_evaluator: GeneralPSEvaluator,
               consistency_threshold: Optional[float] = 0.05,
               delta_fitness_threshold: Optional[float] = 0,
               atomicity_threshold: Optional[float] = 0,
               dependency_threshold: Optional[float] = 0,
               verbose: bool = True) -> list[PS]:
    def filter_by_function(input_pss: list[PS],
                           metric_func: Callable[[PS], float],
                           must_be_below: Optional[float] = None,
                           must_be_above: Optional[float] = None,
                           verbose_name: Optional[str] = "unnamed") -> list[PS]:
        kept = []
        for ps in input_pss:
            metric_value = metric_func(ps)
            if must_be_below is not None:
                if metric_value < must_be_below:
                    kept.append(ps)
                elif verbose:
                    print(f"\t{ps} was rejected because {verbose_name} = {metric_value} > {must_be_below}")
            elif must_be_above is not None:
                if metric_value > must_be_above:
                    kept.append(ps)
                elif verbose:
                    print(f"\t{ps} was rejected because {verbose_name} = {metric_value} < {must_be_above}")
            else:
                raise Exception("The filtering function should specify whether it's above or below")

        return kept

    def maybe_filter_by_consistency(input_pss: list[PS]):
        def get_consistency(ps):
            return -ps.metric_scores[1]

        """ will not filter if consistency threshold is None or the result is empty"""
        if consistency_threshold is not None:
            new_pss = filter_by_function(input_pss, metric_func=get_consistency,
                                         must_be_below=consistency_threshold,
                                         verbose_name="consistency")
            return new_pss if len(new_pss) > 0 else input_pss

    def maybe_filter_by_atomicity(input_pss: list[PS]):
        if verbose:
            print(f"The linkage atomicity threshold is {atomicity_threshold}")

        def get_atomicity(ps):
            return ps.metric_scores[2]

        """ will not filter if consistency threshold is None or the result is empty"""
        if atomicity_threshold is not None:
            new_pss = filter_by_function(input_pss, metric_func=get_atomicity,
                                         must_be_above=atomicity_threshold,
                                         verbose_name="atomicity")
            return new_pss if len(new_pss) > 0 else input_pss

    def maybe_filter_by_dependency(input_pss: list[PS]):
        def get_dependency(ps):
            return -ps.metric_scores[1]

        """ will not filter if consistency threshold is None or the result is empty"""
        if atomicity_threshold is not None:
            new_pss = filter_by_function(input_pss, metric_func=get_dependency,
                                         must_be_below=dependency_threshold,
                                         verbose_name="dependency")
            return new_pss if len(new_pss) > 0 else input_pss

    def maybe_filter_by_delta_fitness(input_pss: list[PS]):
        def get_delta_fitness(ps):
            return ps_evaluator.delta_fitness_metric.get_mean_fitness_delta(ps)

        """ will not filter if consistency threshold is None or the result is empty"""
        if dependency_threshold is not None:
            new_pss = filter_by_function(input_pss, metric_func=get_delta_fitness,
                                         must_be_above=delta_fitness_threshold,
                                         verbose_name="delta_fitness")
            return new_pss if len(new_pss) > 0 else input_pss

    current_pss = pss.copy()

    # current_pss = maybe_filter_by_delta_fitness(current_pss)
    # current_pss = maybe_filter_by_dependency(current_pss)
    current_pss = maybe_filter_by_consistency(current_pss)
    current_pss = maybe_filter_by_atomicity(current_pss)

    return current_pss


def keep_biggest(pss: list[PS]) -> [PS]:
    """returns a singleton list containing the pss with the most variables being fixed, (i know it's counterintuitive"""
    """assumes simplicity is the first metric"""
    return utils.top_with_safe_ties(pss, key=lambda x: np.sum(x.values != STAR), lowest=False)


def keep_with_lowest_dependence(pss: list[PS], local_linkage_metric: TraditionalPerturbationLinkage) -> [PS]:
    return utils.top_with_safe_ties(pss, key=lambda x: local_linkage_metric.get_dependence(x), lowest=True)


def keep_with_best_atomicity(pss: list[PS]) -> [PS]:
    return utils.top_with_safe_ties(pss, key=lambda x: x.metric_scores[2])


def keep_middle(pss: list[PS]) -> [PS]:
    # assuming that they are ordered in a sensible way?
    qty_metrics = len(pss[0].metric_scores)
    metrics = [lambda x: x.metric_scores[i] for i in range(1, qty_metrics)] #
    sorted_pss = utils.sort_by_combination_of(pss, key_functions=metrics)
    middle_index = len(pss) // 2
    return [sorted_pss[middle_index]]


def merge_pss_into_one(pss: list[PS]) -> PS:
    # assumes that no PSS are in disagreement
    pss_matrix = np.array([ps.values for ps in pss])
    final_values = np.max(pss_matrix, axis=0)  # since stars are -1
    return PS(final_values)
