import numpy as np

from Core.PS import STAR, PS
from Core.custom_types import ArrayOfFloats


def distance_between_pss(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    overlap_count = np.sum(np.logical_and(vals_a == vals_b, vals_a != STAR), dtype=float)
    #fixed_count = np.sum(np.logical_or((vals_a != STAR), (vals_b != STAR)), dtype=float)
    fixed_count = np.average((np.sum(vals_a != STAR), np.sum(vals_b != STAR)))

    if fixed_count < 1:
        return 1
    return 1 - (overlap_count / fixed_count)



def sharing_value_between_PSs(vals_a: np.ndarray, vals_b: np.ndarray, sigma_shared: float, alpha: int) -> float:
    distance = distance_between_pss(vals_a, vals_b)
    if distance <= sigma_shared:
        return 1.0 - (distance / sigma_shared)**alpha
    else:
        return 0.0


def get_sharing_score_from_reference_group(solution_matrix_reference: np.ndarray,
                                            ps: PS,
                                            sigma_shared: float, alpha: int) -> float:
    ps_values = ps.values
    penalties = [sharing_value_between_PSs(ps_values, reference_solution, sigma_shared, alpha)
                 for reference_solution in solution_matrix_reference]
    return sum(penalties)

def get_sharing_scores(solution_matrix: np.ndarray, sigma_shared: float, alpha: int) -> ArrayOfFloats:


    solution_amount = len(solution_matrix)
    shared_value_matrix = np.zeros((solution_amount, solution_amount), dtype=float)

    for row_index_a in range(solution_amount):
        row_a = solution_matrix[row_index_a]
        for row_index_b in range(row_index_a+1, solution_amount):
            row_b = solution_matrix[row_index_b]
            shared_value_matrix[row_index_a][row_index_b] = sharing_value_between_PSs(row_a, row_b,
                                                                                      sigma_shared, alpha)

    shared_value_matrix += shared_value_matrix.T
    total_shared_values = np.sum(shared_value_matrix, axis=0)
    return total_shared_values


