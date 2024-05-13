from typing import Any

import numpy as np
from pymoo.core.survival import Survival
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.misc import find_duplicates
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

from Core.PS import STAR


class PyMooCustomCrowding(Survival):
    nds: Any
    filter_out_duplicates: bool

    def __init__(self, nds=None, filter_out_duplicates: bool = True):
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.filter_out_duplicates=filter_out_duplicates


    def __repr__(self):
        return "PyMooCustomGiancarlosCrowding"


    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        raise Exception(f"The class {self.__repr__()} does not implement get_crowding_scores")

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.get_crowding_scores_of_front(
                        F[front, :],
                        n_remove=n_remove,
                        population = pop,
                        front_indexes = front,
                    )

                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                if n_remove != 0:
                    I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.get_crowding_scores_of_front(
                        F[front, :],
                        n_remove=0,
                        population = pop,
                        front_indexes = front,
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]



class PyMooPSGenotypeCrowding(PyMooCustomCrowding):
    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        print("Called PyMooPSGenotypeCrowding.get_crowding_scores_of_front")

        pop_matrix = np.array([ind.X for ind in population])
        where_fixed = pop_matrix != STAR
        counts = np.sum(where_fixed, axis=0)
        foods = (1 / counts).reshape((-1, 1))
        scores = where_fixed[front_indexes] @ foods
        result = scores.ravel()
        colour = "green"

        return result



