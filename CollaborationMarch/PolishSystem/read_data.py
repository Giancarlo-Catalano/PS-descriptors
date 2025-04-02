
#%%
from Core.SearchSpace import SearchSpace
import numpy as np
from Core.PRef import PRef


def get_pRef_from_vectors(name_of_vectors_file: str, name_of_fitness_file: str, column_in_fitness_file: int) -> PRef:
    full_solution_matrix = np.loadtxt(name_of_vectors_file, delimiter=",", dtype=int)
    fitness_array = np.genfromtxt(name_of_fitness_file, delimiter=",", dtype=float, usecols=column_in_fitness_file)
    search_space = SearchSpace(2 for _ in range(full_solution_matrix.shape[1]))
    return PRef(full_solution_matrix=full_solution_matrix,
                fitness_array=fitness_array,
                search_space=search_space)

def get_vectors_file_name(size: int):
    return r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\CollaborationMarch\FromTheirData\data" +f"\\{size}\\" + f"\\many_hot_vectors_{size}_kmeans.csv"


def get_fitness_file_name(size: int):
    return r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\CollaborationMarch\FromTheirData\data" + f"\\{size}\\" + f"\\fitness_{size}_kmeans.csv"



def example_usage_for_read_data():
    size = 20
    fitness_column_to_use = 0

    pRef = get_pRef_from_vectors(name_of_vectors_file=get_vectors_file_name(size),
                                 name_of_fitness_file=get_fitness_file_name(size),
                                 column_in_fitness_file=fitness_column_to_use)
    best_solution = pRef.get_best_solution()

    print(pRef)