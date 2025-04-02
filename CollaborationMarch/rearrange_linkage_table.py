import utils
import numpy as np
def rearrange_table(table, new_order):
    return table[new_order, :][:, new_order]


def find_cleaner_arrangement_with_restriction(table, start_from):
    remaining_rows = list(range(table.shape[0]))

    picked_rows = [start_from]
    remaining_rows.remove(start_from)

    def get_distance_between_rows(row_a, row_b):
        return np.sum(np.abs(table[row_a] - table[row_b]))

    def distance_from_last_row(row):
        return get_distance_between_rows(row, picked_rows[-1])

    while (remaining_rows):
        picked = min(remaining_rows, key=distance_from_last_row)
        picked_rows.append(picked)
        remaining_rows.remove(picked)

    total_penalty = sum(get_distance_between_rows(a, b) for a, b in zip(picked_rows, picked_rows[1:] + [start_from]))

    return picked_rows, total_penalty


def select_best_arrangement_from_many(table):
    all_arrangements = [find_cleaner_arrangement_with_restriction(table, start) for start in range(table.shape[0])]

    best = min(all_arrangements, key=utils.second)
    return best[0]


def clean_table(table):
    best_arrangement = select_best_arrangement_from_many(table)
    return rearrange_table(table, best_arrangement)