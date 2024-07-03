import os
from typing import Iterable, Callable, Any

import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import jit

import utils
from FirstPaper.EvaluatedFS import EvaluatedFS
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import STAR, PS
from FirstPaper.SearchSpace import SearchSpace
from FirstPaper.custom_types import ArrayOfFloats, Fitness


@jit
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

@numba.njit
def get_relevant_rows_in_matrix_shortcircuit(full_solution_matrix, fitness_array, values):
    only_relevant_rows = full_solution_matrix[:, values != STAR]
    only_relevant_values = values[values != STAR]
    return np.array([fitness for fs, fitness in zip(only_relevant_rows, fitness_array)
                     if np.array_equal(fs, only_relevant_values)])


@jit
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

    def __init__(self,
                 fitness_array: Iterable[Fitness],
                 full_solution_matrix: np.ndarray,
                 search_space: SearchSpace):
        self.fitness_array = np.array(fitness_array)
        self.full_solution_matrix = full_solution_matrix
        self.search_space = search_space

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
        fss, fitnesses = utils.unzip([(e_fs.full_solution, e_fs.fitness) for e_fs in evaluated_fss])
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

    def fitnesses_of_observations_experimental(self, ps: PS) -> np.ndarray:
        return get_relevant_rows_in_matrix_shortcircuit(self.full_solution_matrix, self.fitness_array, ps.values)



    def fitnesses_of_observations_other_experimental(self, ps: PS) -> np.ndarray:
        return get_relevant_rows_in_matrix_shortcircuit(self.full_solution_matrix, self.fitness_array, ps.values)
    def fitnesses_of_observations_and_complement(self, ps: PS) -> (ArrayOfFloats, ArrayOfFloats):
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

def plot_solutions_in_pRef(pRef: PRef):
    x_points, y_points = utils.unzip(list(enumerate(pRef.fitness_array)))
    fig = plt.figure()
    plt.plot(x_points, y_points)
    plt.show()

