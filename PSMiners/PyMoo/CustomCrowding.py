from typing import Any

import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

from Core.PS import STAR, PS
from Core.SearchSpace import SearchSpace


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
        #print("Called PyMooPSGenotypeCrowding.get_crowding_scores_of_front")

        pop_matrix = np.array([ind.X for ind in population])
        where_fixed: np.ndarray = pop_matrix != STAR
        counts = np.sum(where_fixed, axis=0)
        foods = (1 / counts).reshape((-1, 1))
        scores = np.array([np.average(foods[row]) for row in where_fixed[front_indexes]])
        return scores


class PyMooPSSequentialCrowding(PyMooCustomCrowding):
    coverage: np.ndarray
    foods: np.ndarray
    search_space: SearchSpace
    opt: Any

    def __init__(self, search_space: SearchSpace, already_obtained: list[PS], immediate = False):
        self.search_space = search_space
        super().__init__()
        self.coverage = PyMooPSSequentialCrowding.get_coverage(self.search_space, already_obtained)
        if immediate:
            self.coverage = np.array([1 if x > 0 else 0 for x in self.coverage])
        self.foods = (1 - self.coverage).reshape((-1, 1))
        self.opt = []


    @classmethod
    def get_coverage(cls, search_space: SearchSpace, already_obtained: list[PS]):
        if len(already_obtained) == 0:
            return np.zeros(search_space.amount_of_parameters, dtype=float)

        pop_matrix = np.array([ps.values for ps in already_obtained])
        where_fixed = pop_matrix != STAR
        counts = np.sum(where_fixed, axis=0)

        return counts / len(already_obtained)


    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        pop_matrix = np.array([population[index].X for index in front_indexes])
        where_fixed: np.ndarray = pop_matrix != STAR

        scores = np.array([np.average(self.foods[row]) if any(row) else 1
                           for row in where_fixed])

        self.opt = population[front_indexes]  # just to comply with Pymoo, ignore this

        return scores


