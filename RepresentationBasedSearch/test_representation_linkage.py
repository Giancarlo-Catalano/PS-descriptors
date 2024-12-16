from BenchmarkProblems.TSP import TSP
from Explanation.PRefManager import PRefManager
from ProblemRepresentation import TrivialRepresentation, CombinedProblemRepresentations
from RepresentationLinkage import RepresentationLinkage
from TSPPredicates import TSPPrecedenceRepresentation, TSPVicinityRepresentation


def test_representation_linkage():
    problem = TSP(
        cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8), (7, 9)],
        starting_ending_city=(5, 5))


    trivial_representation = TrivialRepresentation(problem)
    precedence_representation = TSPPrecedenceRepresentation(problem)
    vicinity_representation = TSPVicinityRepresentation(problem, vicinity_threshold=4)
    representation = CombinedProblemRepresentations([trivial_representation,
                                                     precedence_representation,
                                                     vicinity_representation])


    representation_linkage= RepresentationLinkage(optimisation_problem=problem,
                                                           problem_representation=representation)


    pRef = PRefManager.generate_pRef(problem, sample_size=1000, which_algorithm="GA")
    winning_solution = pRef.get_best_solution()

    representation_linkage.set_solution(winning_solution)

    print("all done I guess...")


test_representation_linkage()
