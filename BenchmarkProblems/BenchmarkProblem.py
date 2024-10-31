from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from utils import announce


class BenchmarkProblem:
    """ This is an interface for toy latest_material, which makes my code much prettier"""
    """ The main components of this class are:
     -  a search space: the combinatorial search space
     -  fitness_function: the fitness function to be MAXIMISED
     -  get_targets: the ideal Core catalog
     -  repr_pr: a way to represent the Core which makes sense for the problem (ie checkerboard would use a grid)
     
     A useful related class to look at is UnitaryProblem
     """
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def __repr__(self):
        raise Exception("An implementation of BenchmarkProblem does not implement __repr__")

    def get_reference_population(self, sample_size: int) -> PRef:
        return PRef.sample_from_search_space(search_space=self.search_space,
                                             fitness_function=self.fitness_function,
                                             amount_of_samples=sample_size)

    def repr_full_solution(self, fs: FullSolution) -> str:
        """default implementation"""
        return f"{fs}"

    def repr_ps(self, ps: PS) -> str:
        """default implementation"""
        return f"{ps}"

    def repr_fs(self, full_solution: FullSolution) -> str:
        return self.repr_ps(PS.from_FS(full_solution))

    def fitness_function(self, fs: FullSolution) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement the fitness function!!!")

    def get_targets(self) -> set[PS]:
        raise Exception("An implementation of BenchmarkProblem does not implement get_targets")

    def get_global_optima_fitness(self) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement get_global_optima_fitness")


    def get_descriptors_of_ps(self, ps: PS) -> dict:
        raise NotImplemented(f"The class {self.__repr__()} does not implement .ps_to_properties")

    def repr_property(self, property_name:str, property_value:str, rank:float, ps: PS):
        start = f"{property_name} = {property_value:.2f} is "


        if rank == 0:
            end = "the lowest observed"
        elif rank == 1.0:
            end = "the highest observed"
        elif rank > 0.5:
            end = f"relatively high (rank = {int(rank * 100)}%)"
        else:
            end = f"relatively low (rank = {int(rank * 100)}%)"

        return start + end


    def repr_extra_ps_info(self, ps: PS):
        return f"PS has {ps.fixed_count()} fixed variables"

    def repr_property_globally(self, k, v, r):
        raise NotImplemented


    def get_readable_property_name(self, property: str) -> str:
        """default behaviour is to just return the raw property name"""
        return property


    def print_stats_of_pss(self, pss: list[PS], full_solutions: list[EvaluatedFS]) -> str:
        raise NotImplemented

    def get_problem_specific_global_information(self, solutions: list[FullSolution]):
        raise NotImplemented


    def print_stats_of_full_solutions(self, param: list[FullSolution]):
        raise NotImplemented

