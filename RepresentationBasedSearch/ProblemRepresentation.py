from typing import TypeAlias, Iterable

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.SearchSpace import SearchSpace

Representation: TypeAlias = FullSolution


class ProblemRepresentation:
    original_problem: BenchmarkProblem
    representation_search_space: SearchSpace

    def __init__(self,
                 original_problem: BenchmarkProblem):
        self.original_problem = original_problem
        self.representation_search_space = self.get_representation_search_space()

    def get_representation_search_space(self) -> SearchSpace:
        raise Exception("Did not implement .get_predicates_search_space")

    def get_representation(self, solution: FullSolution) -> Representation:
        raise Exception("Did not implement .get_predicates")

    def repr_partial_representation(self, partial_representation: PS) -> str:
        raise Exception("Did not implement .repr_partial_representation")

    def repr_representation(self, representation: Representation) -> str:
        return f"{representation}"

    def make_representation_pRef(self, original_pRef: PRef) -> PRef:
        original_solutions = original_pRef.get_evaluated_FSs()
        representation_fsm = np.array([self.get_representation(solution).values for solution in original_solutions])
        return PRef(fitness_array=original_pRef.fitness_array,
                    full_solution_matrix=representation_fsm,
                    search_space=self.representation_search_space)


class TrivialRepresentation(ProblemRepresentation):

    def __init__(self,
                 original_problem: BenchmarkProblem):
        super().__init__(original_problem)

    def get_representation_search_space(self) -> SearchSpace:
        return self.original_problem.search_space

    def get_representation(self, solution: FullSolution) -> Representation:
        return solution

    def repr_partial_representation(self, partial_representation: PS) -> str:
        return self.original_problem.repr_ps(partial_representation)

    def repr_representation(self, representation: Representation) -> str:
        return self.original_problem.repr_fs(representation)


class CombinedProblemRepresentations(ProblemRepresentation):
    representations: list[ProblemRepresentation]
    representation_bounds: list[(int, int)]

    def __init__(self,
                 representations: list[ProblemRepresentation]):
        assert (len(representations) > 0)
        self.representations = representations
        self.representation_bounds = self.get_representations_bounds(r.representation_search_space
                                                                     for r in self.representations)
        super().__init__(representations[0].original_problem)

    @classmethod
    def get_representations_bounds(cls, search_spaces: Iterable[SearchSpace]) -> list[(int, int)]:
        result = []
        starting_index = None
        ending_index = 0
        for s in search_spaces:
            starting_index = ending_index
            ending_index += s.amount_of_parameters
            result.append((starting_index, ending_index))
        return result

    def get_representation_search_space(self) -> SearchSpace:
        return SearchSpace.concatenate_search_spaces(representation.representation_search_space
                                                     for representation in self.representations)

    def get_representation(self, solution: FullSolution) -> Representation:
        values_to_join = [representation.get_representation(solution).values for representation in self.representations]
        values = np.hstack(values_to_join)
        return FullSolution(values)

    def break_into_representations(self, representation: Representation) -> list[Representation]:
        return [FullSolution(representation.values[start:end]) for start, end in self.representation_bounds]

    def break_into_partial_representations(self, partial_representation: PS) -> list[PS]:
        return [PS(partial_representation.values[start:end]) for start, end in self.representation_bounds]

    def repr_partial_representation(self, partial_representation: PS) -> str:
        sub_reprs = self.break_into_partial_representations(partial_representation)
        # losing my mind as I type
        return "\n".join(r.repr_partial_representation(item) for r, item in zip(self.representations, sub_reprs))

    def repr_representation(self, representation: Representation) -> str:
        as_ps = PS.from_FS(representation)
        return self.repr_partial_representation(as_ps)
