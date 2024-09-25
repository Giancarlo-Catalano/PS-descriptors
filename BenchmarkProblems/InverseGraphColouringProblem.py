import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.SearchSpace import SearchSpace


class InverseGraphColouring(BenchmarkProblem):
    amount_of_colours: int
    clique_size: int
    amount_of_cliques: int

    def __init__(self,
                 amount_of_colours: int,
                 clique_size: int,
                 amount_of_cliques: int):
        self.amount_of_colours = amount_of_colours
        self.clique_size = clique_size
        self.amount_of_cliques = amount_of_cliques

        search_space = SearchSpace([amount_of_colours for _ in range(self.clique_size * self.amount_of_cliques)])
        super().__init__(search_space)

    def __repr__(self):
        return f"InverseGraphColouring(#colours = {self.amount_of_colours}, #cliques = {self.amount_of_cliques}, clique_size = {self.clique_size})"

    def fitness_function(self, fs: FullSolution) -> float:
        raise Exception("Trying to use InverseGraphColouringProblem directly!!!!!")


    def generate_targets(self) -> set[PS]:
        all_stars = np.full(fill_value=-1, shape=self.search_space.amount_of_parameters)

        def make_ps_for_clique_and_colour(clique_index: int, colour: int) -> PS:
            result_values = all_stars.copy()
            result_values[clique_index*self.clique_size:(clique_index+1)*self.clique_size] = colour
            return PS(result_values)

        return {make_ps_for_clique_and_colour(clique_index, colour)
                for clique_index in range(self.amount_of_cliques)
                for colour in range(self.amount_of_colours)}

    def get_targets(self) -> set[PS]:
        return self.generate_targets()
