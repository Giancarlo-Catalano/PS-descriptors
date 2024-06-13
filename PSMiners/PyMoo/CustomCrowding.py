from typing import Any, Iterable

import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort

import utils
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
                if n_remove != 0:  # otherwise we get a bug in the normal implementation!!!
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
        #self.foods = (1 - self.coverage).reshape((-1, 1))
        self.foods = 1/((self.coverage * len(already_obtained))+1).reshape((-1, 1))
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



def distance_between_pss(ps_a: PS, ps_b: PS) -> float:
    vals_a = ps_a.values
    vals_b = ps_b.values
    overlap_count = np.sum(np.logical_and(vals_a == vals_b, vals_a != STAR), dtype=float)
    #fixed_count = np.sum(np.logical_or((vals_a != STAR), (vals_b != STAR)), dtype=float)
    fixed_count = np.average((np.sum(vals_a != STAR), np.sum(vals_b != STAR)))

    if fixed_count < 1:
        return 1
    return 1 - (overlap_count / fixed_count)

class PyMooDecisionSpaceSequentialCrowding(PyMooCustomCrowding):
    """ This is the one!!!!!"""
    archived_pss: set[PS]
    sigma_shared: float
    opt: Any

    def __init__(self, archived_pss: Iterable[PS], sigma_shared: float):
        super().__init__()
        self.archived_pss = set(archived_pss)
        self.sigma_shared = sigma_shared
        self.opt = []


    def is_too_close(self, ps_a: PS, ps_b: PS) -> bool:
        return distance_between_pss(ps_a, ps_b) < self.sigma_shared

    def get_crowding_score(self, ps: PS) -> float:
        if len(self.archived_pss) == 0:
            return 1
        amount_of_close = len([archived for archived in self.archived_pss if self.is_too_close(ps, archived)])
        return 1 - (amount_of_close / len(self.archived_pss))


    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        pss  = [PS(population[index].X) for index in front_indexes]
        scores = np.array([self.get_crowding_score(ps) for ps in pss])

        self.opt = population[front_indexes]  # just to comply with Pymoo, ignore this
        return scores




class AggressivePyMooPSSequentialCrowding(PyMooCustomCrowding):  # just for debu
    search_space: SearchSpace
    elite: list[PS]
    opt: Any

    def __init__(self, search_space: SearchSpace, already_obtained: list[PS], immediate = False):
        self.search_space = search_space
        super().__init__()
        self.elite  = already_obtained
        self.opt = []

    @classmethod  # for legacy reasons
    def get_coverage(cls, search_space: SearchSpace, already_obtained: list[PS]):
        if len(already_obtained) == 0:
            return np.zeros(search_space.amount_of_parameters, dtype=float)

        pop_matrix = np.array([ps.values for ps in already_obtained])
        where_fixed = pop_matrix != STAR
        counts = np.sum(where_fixed, axis=0)

        return counts / len(already_obtained)



    def get_offence_score_for_ps(self, ps: PS) -> int:

        def is_offended(from_elite: PS) -> bool:
            return any((from_elite.values != STAR) & (ps.values != STAR))

        return len([from_elite for from_elite in self.elite if is_offended(from_elite)])


    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        # note to self: 1 is good, 0 is bad
        pss = [PS(individual.X) for individual in population[front_indexes]]
        offence_scores: list[int] = [self.get_offence_score_for_ps(ps) for ps in pss]
        final_scores = utils.remap_array_in_zero_one(-np.array(offence_scores))
        self.opt = population[front_indexes]  # just to comply with Pymoo, ignore this

        return final_scores





