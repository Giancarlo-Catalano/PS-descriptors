from typing import Any

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.sms import SMSEMOA
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
    """This is dogshit"""
    n_params = search_space.amount_of_parameters
    def get_ref_dirs():
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
        return ref_dirs

    def get_ref_points():
        return np.array([[0, 0.6, 1]])


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
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return MOEAD(ref_dirs = get_ref_dirs(), sampling=sampling, crossover=crossover,
            mutation=mutation, n_neighbors=n_params, survival = crowding_operator)
        else:
            return MOEAD(ref_dirs = get_ref_dirs(), sampling=sampling, crossover=crossover,
                         mutation=mutation, n_neighbors=n_params)
    elif which_algorithm == "RVEA":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return RVEA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                        mutation=mutation, eliminate_duplicates=True,
                        ref_dirs=get_ref_dirs(), survival = crowding_operator)
        else:
            return RVEA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                        mutation=mutation, eliminate_duplicates=True,
                        ref_dirs=get_ref_dirs())
    elif which_algorithm == "SPEA2":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                         mutation=mutation, eliminate_duplicates=True,
                         ref_dirs=get_ref_dirs(), survival = crowding_operator)
        else:
            return SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                        mutation=mutation, eliminate_duplicates=True,
                        ref_dirs=get_ref_dirs())
    elif which_algorithm == "R-NSGA-III":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return RNSGA3(pop_size=pop_size, sampling=sampling, crossover=crossover,
                          mutation=mutation, eliminate_duplicates=True,
                          ref_points = get_ref_points(),
                          pop_per_ref_point=pop_size//5, survival = crowding_operator)
        else:
            return RNSGA3(pop_size=pop_size, sampling=sampling, crossover=crossover,
                        mutation=mutation, eliminate_duplicates=True,
                          ref_points = get_ref_points(),
                          pop_per_ref_point=pop_size//5)
    elif which_algorithm == "SMS-EMOA":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return SMSEMOA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                           mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
        else:
            return SMSEMOA(pop_size=pop_size, sampling=sampling, crossover=crossover,
                        mutation=mutation, eliminate_duplicates=True)
    elif which_algorithm == "R-NSGA-II":
        if isinstance(crowding_operator, PyMooCustomCrowding):
            return RNSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                            mutation=mutation, eliminate_duplicates=True,
                              ref_points = get_reference_directions("das-dennis", 3, n_partitions=7),
                              epsilon=0.1,
                              normalization='front',
                              extreme_points_as_reference_points=False,
                              weights=[1/3, 1/3, 1/3], survival = crowding_operator)
        else:
            return RNSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                          mutation=mutation, eliminate_duplicates=True,
                          ref_points = get_reference_directions("das-dennis", 3, n_partitions=7),
                          epsilon=0.1,
                          normalization='front',
                          extreme_points_as_reference_points=False,
                          weights=[1/3, 1/3, 1/3])
    else:
        raise Exception(f"The algorithm {which_algorithm} was not recognised...")