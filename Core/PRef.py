import heapq
import random
from typing import Iterable, Callable, Any

import numpy as np
from matplotlib import pyplot as plt

import utils
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import STAR, PS
from Core.SearchSpace import SearchSpace
from Core.custom_types import ArrayOfFloats, Fitness


#@jit
def get_relevant_rows_in_matrix_old(fs_matrix, fs_fitnesses, ps_values) -> np.ndarray:
    only_relevant_rows = fs_matrix[:, ps_values != STAR]
    only_relevant_values = ps_values[ps_values != STAR]

    # Rewrite np.all() to use np.alltrue() for Numba compatibility
    def alltrue(arr):
        for elem in arr:
            if not elem:
                return False
        return True

    where_matching = np.array([alltrue(row == only_relevant_values) for row in only_relevant_rows])
    return fs_fitnesses[where_matching]

#@numba.njit
def get_relevant_rows_in_matrix_shortcircuit(full_solution_matrix, fitness_array, values):
    only_relevant_rows = full_solution_matrix[:, values != STAR]
    only_relevant_values = values[values != STAR]
    return np.array([fitness for fs, fitness in zip(only_relevant_rows, fitness_array)
                     if np.array_equal(fs, only_relevant_values)])


#@jit
def get_relevant_rows_in_matrix(fs_matrix, fs_fitnesses, ps_values) -> np.ndarray:
    # Find the indices where ps_values is not equal to STAR
    relevant_indices = np.where(ps_values != STAR)[0]

    # Extract only the relevant values from ps_values
    relevant_values = ps_values[relevant_indices]

    # Extract only the relevant columns from fs_matrix
    relevant_matrix = fs_matrix[:, relevant_indices]

    # Initialize an empty list to store matching row indices
    matching_rows = []

    # Iterate over rows of the relevant matrix
    for row in relevant_matrix:
        # Initialize a flag to track whether the row matches the relevant values
        match = True

        # Iterate over elements of the row and relevant values simultaneously
        for elem, val in zip(row, relevant_values):
            if elem != val:
                match = False
                break  # Exit inner loop early if mismatch is found

        # If match is still True after iterating over all elements, add the row index to matching_rows
        matching_rows.append(match)

    # Convert matching_rows to a NumPy array
    matching_rows = np.array(matching_rows)

    # Return the corresponding fitness values for the matching rows
    return fs_fitnesses[matching_rows]


class PRef:
    """
    This class represents the referenece population, and you should think of it as a list of solutions,
    and a list of their fitnesses. Everything else is just to make the calculations faster / easier to implement.
    """
    fitness_array: ArrayOfFloats
    full_solution_matrix: np.ndarray
    search_space: SearchSpace

    cached_mean: float

    def __init__(self,
                 fitness_array: Iterable[Fitness],
                 full_solution_matrix: np.ndarray,
                 search_space: SearchSpace):
        self.fitness_array = np.array(fitness_array)
        self.full_solution_matrix = full_solution_matrix
        self.search_space = search_space
        self.cached_mean = np.average(self.fitness_array)

    def __repr__(self):
        mean_fitness = np.average(self.fitness_array)

        return f"PRef with {self.sample_size} samples, mean = {mean_fitness:.2f}"

    @classmethod
    def from_full_solutions(cls, full_solutions: Iterable[FullSolution],
                            fitness_values: Iterable[Fitness],
                            search_space: SearchSpace):
        matrix = np.array([fs.values for fs in full_solutions])
        return cls(fitness_values, matrix, search_space)


    @classmethod
    def from_evaluated_full_solutions(cls, evaluated_fss: Iterable[EvaluatedFS],
                                      search_space: SearchSpace):
        fss, fitnesses = utils.unzip([(e_fs, e_fs.fitness) for e_fs in evaluated_fss])
        return cls.from_full_solutions(fss, fitnesses, search_space)

    @classmethod
    def sample_from_search_space(cls, search_space: SearchSpace,
                                 fitness_function: Callable,
                                 amount_of_samples: int):
        samples = [FullSolution.random(search_space) for _ in range(amount_of_samples)]
        fitnesses = [fitness_function(fs) for fs in samples]
        return cls.from_full_solutions(samples, fitnesses, search_space)


    def fitnesses_of_observations(self, ps: PS) -> ArrayOfFloats:
        """
        This is the most important function of the class, and it roughly corresponds to the obs_PRef(ps) in the paper
        :param ps: a partial solution, where the * values are represented by -1
        :return: a list of floats, corresponding to the fitnesses of the observations of the ps
        within the reference population
        """
        remaining_rows = self.full_solution_matrix
        remaining_fitnesses = self.fitness_array

        for variable_index, variable_value in enumerate(ps.values):
            if variable_value != STAR:
                which_to_keep = remaining_rows[:, variable_index] == variable_value

                # update the current filtered results
                remaining_rows = remaining_rows[which_to_keep]
                remaining_fitnesses = remaining_fitnesses[which_to_keep]

        return remaining_fitnesses

    def get_indexes_matching_ps(self, ps: PS) -> np.ndarray:

        remaining_indexes = np.arange(self.sample_size)

        for var, val in enumerate(ps.values):
            if val != STAR:
                subset_that_matches = self.full_solution_matrix[remaining_indexes][:, var] == val
                remaining_indexes = remaining_indexes[subset_that_matches]

        return remaining_indexes


    def fitnesses_of_observations_experimental(self, ps: PS) -> np.ndarray:
        return get_relevant_rows_in_matrix_shortcircuit(self.full_solution_matrix, self.fitness_array, ps.values)



    def fitnesses_of_observations_other_experimental(self, ps: PS) -> np.ndarray:
        return get_relevant_rows_in_matrix_shortcircuit(self.full_solution_matrix, self.fitness_array, ps.values)
    def fitnesses_of_observations_and_complement(self, ps: PS) -> (ArrayOfFloats, ArrayOfFloats):
        """Returns the fitnesses for the rows where the ps is present,
        and the fitnesses for the rows where it's not present"""
        selected_rows = np.full(shape=self.fitness_array.shape, fill_value=True, dtype=bool)

        for variable_index, variable_value in enumerate(ps.values):
            if variable_value != STAR:
                rows_where_variable_matches = self.full_solution_matrix[:, variable_index] == variable_value
                selected_rows = np.logical_and(selected_rows, rows_where_variable_matches)

        return self.fitness_array[selected_rows], self.fitness_array[np.logical_not(selected_rows)]

    @property
    def sample_size(self) -> int:
        return len(self.fitness_array)

    def get_with_normalised_fitnesses(self):
        normalised_fitnesses = utils.remap_array_in_zero_one(self.fitness_array)
        return PRef(fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=self.full_solution_matrix,
                    search_space=self.search_space)

    def get_fitnesses_matching_var_val(self, var: int, val: int) -> ArrayOfFloats:
        where = self.full_solution_matrix[:, var] == val
        return self.fitness_array[where]

    def get_fitnesses_matching_var_val_pair(self, var_a: int, val_a: int, var_b: int, val_b: int) -> ArrayOfFloats:
        where = np.logical_and(self.full_solution_matrix[:, var_a] == val_a,
                               self.full_solution_matrix[:, var_b] == val_b)
        return self.fitness_array[where]

    def get_evaluated_FSs(self) -> list[EvaluatedFS]:
        return [EvaluatedFS(full_solution=FullSolution(row), fitness=fitness) for row, fitness in
                zip(self.full_solution_matrix, self.fitness_array)]

    def describe_self(self):
        min_fitness = np.min(self.fitness_array)
        max_fitness = np.max(self.fitness_array)
        avg_fitness = np.average(self.fitness_array)
        print(
            f"This PRef contains {self.sample_size} samples, where the minimum is {min_fitness}, the maximum = {max_fitness} and the average is {avg_fitness}")

    def save(self, file, verbose=False):
        # create the folder if it doesn't exist
        utils.make_folder_if_not_present(file)
        np.savez(file,
                 fsm=self.full_solution_matrix,
                 fitness_array=self.fitness_array,
                 search_space=self.search_space.cardinalities)


    @classmethod
    def load(cls, file: str):
        results = np.load(file)
        return cls(full_solution_matrix=results["fsm"],
                   fitness_array=results["fitness_array"],
                   search_space=SearchSpace(results["search_space"]))



    @classmethod
    def concat(cls, pRefs: list[Any]):
        if len(pRefs) == 0:
            raise Exception("Cannot concatenate 0 pRefs")
        elif len(pRefs) == 1:
            return pRefs[0]
        else:
            fsm = np.vstack(tuple(pRef.full_solution_matrix for pRef in pRefs))
            fitness_array = np.concatenate([pRef.fitness_array for pRef in pRefs])
            search_space = pRefs[0].search_space
            return cls(full_solution_matrix=fsm,
                       fitness_array = fitness_array,
                       search_space = search_space)

    @classmethod
    def unique(cls, pRef):
        """ removes duplicate entries"""
        all_solutions = pRef.get_evaluated_FSs()
        all_solutions = list(set(all_solutions))
        return PRef.from_evaluated_full_solutions(all_solutions, pRef.search_space)

    def get_random_evaluated_fs(self) -> EvaluatedFS:
        index = random.randrange(self.sample_size)
        return EvaluatedFS(FullSolution(self.full_solution_matrix[index]),
                           fitness=self.fitness_array[index])

    def get_nth_solution(self, index: int) -> EvaluatedFS:
        return EvaluatedFS(FullSolution(self.full_solution_matrix[index]),
                           fitness=self.fitness_array[index])


    def get_top_n_solutions(self, n: int) -> list[EvaluatedFS]:
        indexes_and_fitnesses = list(enumerate(self.fitness_array))
        best_indexes_and_fitnesses = heapq.nlargest(n=n, iterable=indexes_and_fitnesses, key=utils.second)

        return [self.get_nth_solution(index) for index, _ in best_indexes_and_fitnesses]

    def get_best_solution(self) -> EvaluatedFS:
        best_index: int = np.argmax(self.fitness_array)
        return self.get_nth_solution(best_index)

    def get_sorted(self, reverse = True) -> Any:  # returns a pRef
        enumerated_fitnesses = sorted(enumerate(self.fitness_array), key=utils.second, reverse=reverse)
        new_indexes, new_fitnesses = zip(*enumerated_fitnesses)
        new_indexes = np.array(new_indexes) ## otherwise the indexing doesn't work??
        new_fitnesses = np.array(new_fitnesses)
        new_full_solution_matrix = self.full_solution_matrix[new_indexes]
        return PRef(fitness_array=new_fitnesses,
                    full_solution_matrix=new_full_solution_matrix,
                    search_space=self.search_space)


    def split_by_indexes(self, indexes_in_match: Iterable[int]) -> (Any, Any):
        not_matches = np.ones(shape=self.fitness_array.shape, dtype=bool)
        not_matches[indexes_in_match] = False
        not_matching_indexes = np.arange(self.sample_size)[not_matches]

        matching_pRef = PRef(fitness_array=self.fitness_array[indexes_in_match],
                             full_solution_matrix=self.full_solution_matrix[indexes_in_match],
                             search_space=self.search_space)

        not_matching_pRef = PRef(fitness_array=self.fitness_array[not_matching_indexes],
                                 full_solution_matrix=self.full_solution_matrix[not_matching_indexes],
                                 search_space=self.search_space)
        return matching_pRef, not_matching_pRef


    def train_test_split(self, test_size: float, random_state: int = None) -> (Any, Any):
        if random_state is not None:
            random.seed(random_state)
        test_indexes = random.sample(range(self.sample_size), int(self.sample_size * test_size))
        test, train = self.split_by_indexes(test_indexes)
        return train, test  #  had to do this to flip them


def plot_solutions_in_pRef(pRef: PRef, filename: str):
    x_points, y_points = utils.unzip(list(enumerate(pRef.fitness_array)))
    fig = plt.figure()
    plt.plot(x_points, y_points)
    plt.show()
    #plt.savefig(filename)
    plt.close(fig)

