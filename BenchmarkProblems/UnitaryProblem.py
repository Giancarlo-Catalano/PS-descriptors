import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from Core.custom_types import ArrayOfInts


class UnitaryProblem(BenchmarkProblem):
    """ This interface represents all latest_material where """
    amount_of_cliques: int
    clique_size: int

    def __init__(self,
                 amount_of_cliques: int,
                 clique_size: int):
        self.amount_of_cliques = amount_of_cliques
        self.clique_size = clique_size
        search_space = SearchSpace([2 for _ in range(self.amount_of_bits)])
        super().__init__(search_space)

    def get_bit_counts(self, full_solution: FullSolution) -> ArrayOfInts:
        bits = full_solution.values.reshape((-1, self.clique_size))
        return np.sum(bits, axis=1)

    @property
    def amount_of_bits(self) -> int:
        return self.amount_of_cliques * self.clique_size

    @staticmethod
    def get_optimal_clique(clique_size: int) -> FullSolution:
        raise Exception(f"An implementation of UnitaryProblem not implement get_optimal_clique")

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        raise Exception("An implementation of UnitaryProblem does not implement .unitary_function")

    def get_optimal_fitness_per_clique(self) -> float:
        optimal_clique = self.get_optimal_clique(self.clique_size)
        bitcount = optimal_clique.values.sum()
        return self.unitary_function(bitcount, self.clique_size)

    def get_global_optima_fitness(self) -> float:
        return self.get_optimal_fitness_per_clique() * self.amount_of_cliques

    def fitness_function(self, full_solution: FullSolution) -> float:
        return sum(self.unitary_function(bc, self.clique_size) for bc in self.get_bit_counts(full_solution))

    def get_problem_name(self) -> str:
        raise Exception(
            "An implementation of UnitaryProblem does not implement get_problem_name, which is used in __repr__")

    def __repr__(self):
        return f"{self.get_problem_name()}(#cliques = {self.amount_of_cliques}, size = {self.clique_size})"

    def repr_ps(self, ps: PS) -> str:
        def repr_cell(cell_value: int) -> str:
            match cell_value:
                case 0:
                    return "0"
                case 1:
                    return "1"
                case _:
                    return "*"

        def repr_clique(clique: np.ndarray) -> str:
            return f'{" ".join(repr_cell(cell) for cell in clique)}'

        cliques = ps.values.reshape((-1, self.clique_size))
        return "["+"  ".join(repr_clique(clique) for clique in cliques)+"]"

    def repr_fs(self, full_solution: FullSolution) -> str:
        as_ps = PS.from_FS(full_solution)
        return self.repr_ps(as_ps)

    def get_targets(self) -> list[PS]:
        optimal_clique_values = self.get_optimal_clique(self.clique_size).values

        all_star_values = np.full(shape=self.amount_of_bits, fill_value=STAR)

        def set_at_clique(clique_number: int) -> PS:
            result_values = all_star_values.copy()
            clique_start = clique_number * self.clique_size
            clique_end = (clique_number + 1) * self.clique_size
            result_values[clique_start:clique_end] = optimal_clique_values
            return PS(result_values)

        return [set_at_clique(which) for which in range(self.amount_of_cliques)]
