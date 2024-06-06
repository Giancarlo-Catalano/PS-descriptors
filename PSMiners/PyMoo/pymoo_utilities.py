from typing import Any

import numpy as np
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.survival import Survival
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

import utils
from Core.SearchSpace import SearchSpace
from PSMiners.PyMoo.CustomCrowding import PyMooCustomCrowding



def tournament_select_for_pymoo(pop, P, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    S = np.full(n_tournaments, -1, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        indexes = P[i]
        fs = [pop[i].F for i in indexes]
        indexes_and_fs = list(zip(indexes, fs))
        indexes_and_fs.sort(key=utils.second)
        S[i] =  indexes[0][0]
    return S


def get_pymoo_search_algorithm(which_algorithm: str,
                               search_space: SearchSpace,
                               pop_size: int,
                               sampling: Any,
                               crowding_operator: Survival,
                               crossover: Any,
                               mutation: Any):
    n_params = search_space.amount_of_parameters
    def get_ref_dirs():
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        return (ref_dirs + 1) / 3
    if which_algorithm == "NSGAII":
        return NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                      mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
    if which_algorithm == "NSGAIII":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return NSGA3(pop_size=pop_size, ref_dirs=get_ref_dirs(), sampling=sampling,
                         crossover=crossover, mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
        else:
            return NSGA3(pop_size=pop_size, ref_dirs=get_ref_dirs(), sampling=sampling,
                         crossover=crossover, mutation=mutation, eliminate_duplicates=True)
    elif which_algorithm == "MOEAD":
        return MOEAD(ref_dirs = get_ref_dirs(), sampling=sampling, crossover=crossover,
            mutation=mutation, n_neighbors=n_params, prob_neighbor_mating=0.7,
            survival=crowding_operator)
    elif which_algorithm == "AGEMOEA":
        return AGEMOEA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                       mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
    elif which_algorithm == "RVEA":
        return RVEA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                    mutation=mutation, eliminate_duplicates=True, survival=crowding_operator,
                    ref_dirs=get_ref_dirs())
    elif which_algorithm == "SPEA2":
        return SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                    mutation=mutation, eliminate_duplicates=True, survival=crowding_operator,
                    ref_dirs=get_ref_dirs())
    else:
        raise Exception(f"The algorithm {which_algorithm} was not recognised...")