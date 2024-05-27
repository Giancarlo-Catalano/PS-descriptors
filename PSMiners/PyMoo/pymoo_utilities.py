from typing import Any

from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.survival import Survival
from pymoo.util.ref_dirs import get_reference_directions

from Core.SearchSpace import SearchSpace


def get_pymoo_search_algorithm(which_algorithm: str,
                               search_space: SearchSpace,
                               pop_size: int,
                               sampling: Any,
                               crowding_operator: Survival,
                               crossover: Any,
                               mutation: Any):
    n_params = search_space.amount_of_parameters
    def get_ref_dirs():
        return get_reference_directions("das-dennis", 3, n_partitions=12)
    if which_algorithm == "NSGAII":
        return NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover,
                      mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
    if which_algorithm == "NSGAIII":
        return NSGA3(pop_size=pop_size, ref_dirs=get_ref_dirs(), sampling=sampling,
                     crossover=crossover, mutation=mutation, eliminate_duplicates=True, survival=crowding_operator)
    elif which_algorithm == "MOEAD":
        return MOEAD(ref_dirs = get_ref_dirs(), sampling=sampling, crossover=crossover,
            mutation=mutation, n_neighbors=n_params, prob_neighbor_mating=0.7,
            survival=crowding_operator
        )
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