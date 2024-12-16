import heapq
import random

import numpy as np

from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class FSMutationOperator:
    search_space: SearchSpace


    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space


    def __repr__(self) -> str:
        raise Exception("A FS mutation operator does not implement __repr__")


    def mutated(self, fs: FullSolution) -> FullSolution:
        raise Exception(f"The class {self.__repr__()} does not implement .mutated")


class SinglePointFSMutation(FSMutationOperator):
    probability: float
    def __init__(self, search_space: SearchSpace, probability = None):
        super().__init__(search_space)
        if probability is None:
            self.probability = 1/self.search_space.amount_of_parameters
        else:
            self.probability = probability


    def mutated(self, fs: FullSolution) -> FullSolution:
        new_values = fs.values.copy()
        for index, _ in enumerate(fs.values):
            if random.random() < self.probability:
                new_values[index] = random.randrange(self.search_space.cardinalities[index])
        return FullSolution(new_values)


    def __repr__(self):
        return "SinglePointFSMutation"




class FSCrossoverOperator:
    def __init__(self):
        pass

    def __repr__(self):
        raise Exception("An implementation of FSCrossoveroperator does not implement __repr__")


    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        raise Exception(f"The class {self.__repr__()} does not implement .crossed")


class TwoPointFSCrossover(FSCrossoverOperator):
    def __init__(self):
        super().__init__()

    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        last_index = len(mother)
        start_cut = random.randrange(last_index)
        end_cut = random.randrange(last_index)
        start_cut, end_cut = min(start_cut, end_cut), max(start_cut, end_cut)

        def take_from(donor: FullSolution, start_index, end_index) -> list[int]:
            return list(donor.values[start_index:end_index])

        child_value_list = (take_from(mother, 0, start_cut) +
                            take_from(father, start_cut, end_cut) +
                            take_from(mother, end_cut, last_index))

        return FullSolution(child_value_list)

    def __repr__(self):
        return "TwoPointFSCrossover"


class UniformFSCrossover(FSCrossoverOperator):
    def __init__(self):
        super().__init__()


    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        result_values = mother.values.copy()
        n = len(result_values)
        values_to_borrow = np.random.randint(2, size=n, dtype=bool)
        result_values[values_to_borrow] = father.values[values_to_borrow]
        return FullSolution(result_values)


    def __repr__(self):
        return "UniformFSCrossover"



class FSSelectionOperator:
    def __init__(self):
        pass

    def __repr__(self):
        raise Exception("An implementation of FSSelectionOperator does not implement __repr__")

    def select(self, population: list[FullSolution], amount: int) -> list[FullSolution]:
        raise Exception(f"The class {self.__repr__()} does not implement .select")




class TournamentSelection(FSSelectionOperator):
    tournament_size: int


    def __init__(self, tournament_size = 3):
        super().__init__()
        self.tournament_size = tournament_size


    def __repr__(self):
        return "FSTournamentSelection"



    def select_single(self, population: list[FullSolution]) -> FullSolution:
        return max(random.choices(population, k=self.tournament_size))

    def select(self, population: list[FullSolution], amount: int) -> list[FullSolution]:
        return [self.select_single(population) for _ in range(amount)]


class TruncationSelection(FSSelectionOperator):
    def __init__(self):
        super().__init__()


    def __repr__(self):
        return "FSTruncationSelection"

    def select(self, population: list[FullSolution], amount: int) -> list[FullSolution]:
        return heapq.nlargest(n=amount, iterable=population)


