import random
from typing import Any

import numpy as np

from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from utils import announce


# due to my Python being old, I'm forced to use Any instead of Self

def safe_average(fitness_array: np.ndarray) -> float:
    if len(fitness_array) == 0:
        return -10000
    else:
        return np.average(fitness_array)

class CachedRowsNode:
    remaining_solution_matrix: np.ndarray
    remaining_fitnesses: np.ndarray

    cached_mean_fitness: float

    branches: list[(int, int), Any]  # var -> val -> node



    def __init__(self,
                 remaining_solution_matrix: np.ndarray,
                 remaining_fitnesses: np.ndarray,
                 cached_mean_fitness: float,
                 branches: list[(int, int), Any]):
        self.remaining_solution_matrix = remaining_solution_matrix
        self.remaining_fitnesses = remaining_fitnesses
        self.cached_mean_fitness = cached_mean_fitness
        self.branches = branches

    @classmethod
    def root_node_from_pRef(cls, pRef: PRef) -> Any:
        return cls(remaining_solution_matrix = pRef.full_solution_matrix,
                   remaining_fitnesses = pRef.fitness_array,
                   cached_mean_fitness = safe_average(pRef.fitness_array),
                   branches = [])

    @classmethod
    def specialised_from(cls, node: Any, new_var: int, new_val: int) -> Any:
        which_rows_remain = node.remaining_solution_matrix[:, new_var] == new_val
        remaining_rows = node.remaining_solution_matrix[which_rows_remain]
        remaining_fitnesses = node.remaining_fitnesses[which_rows_remain]
        cached_mean_fitness = safe_average(remaining_fitnesses)
        return cls(remaining_solution_matrix=remaining_rows,
                   remaining_fitnesses = remaining_fitnesses,
                   cached_mean_fitness = cached_mean_fitness,
                   branches=[])
    def add_sub_branch(self, var: int, val: int) -> Any:
        new_node = CachedRowsNode.specialised_from(self, var, val)
        self.branches.append(((var, val), new_node))

        return new_node


    def navigate_to_next_branch(self, available_var_vals: set[(int, int)]): #  -> Optional[((int, int), Any)]
        # attempts to go deeper in the tree
        # if successfull, it tells you which branch it took (varval), and the new branch (branch) -> (varval, branch)
        # otherwise it returns None

        for (var_val, sub_branch) in self.branches:
            if var_val in available_var_vals:
                return (var_val, sub_branch)

        # finally
        return None




def fast_get_mean_fitness(ps: PS, solution_row_tree_root: CachedRowsNode) -> float:
    # this function will explore the tree to find the cached mean fitness if possible, or update the tree to match

    remaining_var_val_pairs = set((var, val) for (var, val) in enumerate(ps.values) if val != STAR)
    current_node = solution_row_tree_root

    while True:
        # try to navigate to a sub-branch
        maybe_next_branch = current_node.navigate_to_next_branch(remaining_var_val_pairs)
        if maybe_next_branch is None:
            break # terminate when none are available
        else:
            var_val_to_remove, current_node = maybe_next_branch
            remaining_var_val_pairs.remove(var_val_to_remove)

    # add as many branches as needed
    while len(remaining_var_val_pairs) > 0:
        branch_to_add = remaining_var_val_pairs.pop()
        current_node = current_node.add_sub_branch(*branch_to_add)

    return current_node.cached_mean_fitness



def test_solution_row_cacher():
    problem = RoyalRoad(4, 4)
    pRef = problem.get_reference_population(sample_size=10000)

    mean_fitness_metric = MeanFitness()
    mean_fitness_metric.set_pRef(pRef)


    root_node = CachedRowsNode.root_node_from_pRef(pRef)

    def random_PS(amount_of_fixed: int) -> PS:
        result = PS.empty(problem.search_space)
        for _ in range(amount_of_fixed):
            new_var = random.randrange(problem.search_space.amount_of_parameters)
            new_val = random.randrange(problem.search_space.cardinalities[new_var])
            result = result.with_fixed_value(new_var, new_val)

        return result


    amount_of_samples = 100000

    random_pss = [random_PS(random.randrange(6)) for _ in range(amount_of_samples)]

    with announce("doing it the traditional way"):
        mean_fitnesses = [mean_fitness_metric.get_single_score(ps) for ps in random_pss]

    with announce("doing it the novel way"):
        mean_fitnesses = [fast_get_mean_fitness(ps, root_node) for ps in random_pss]


    # for amount_fixed_vars in range(24):
    #     for _ in range(12):
    #         ps = random_PS(amount_fixed_vars)
    #         from_trad = mean_fitness_metric.get_single_score(ps)
    #         from_novel = fast_get_mean_fitness(ps, root_node)
    #         difference = abs(from_trad - from_novel)
    #         print(f"For {ps}, {difference = :.2f}")


    print("All done")






