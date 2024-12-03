import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, contains
from Core.TerminationCriteria import FullSolutionEvaluationLimit
from FSStochasticSearch.GA import GA
from FSStochasticSearch.Operators import UniformFSCrossover, SinglePointFSMutation, TournamentSelection
from ProblemRepresentation import ProblemRepresentation, Representation


def find_solution_with_closest_representation(target_representation: FullSolution,
                                              parts_that_have_to_match: PS,
                                              original_problem: BenchmarkProblem,
                                              representation: ProblemRepresentation,
                                              search_budget: int,
                                              population_size) -> (FullSolution, Representation):
    # it's a GA task

    def distance_metric_between_representations(repr_a: Representation, repr_b: Representation) -> float:
        # manhattan distance basically
        differences = repr_a.values != repr_b.values
        return float(np.sum(differences, dtype=float))

    def satisfies_forced_requirement(representation: Representation) -> bool:
        return contains(representation, parts_that_have_to_match)

    def fitness_function(fs: FullSolution) -> float:
        # the GA is a maximisation thing, so we need to make it negative
        fs_representation = representation.get_representation(fs)
        if satisfies_forced_requirement(fs_representation):
            return -distance_metric_between_representations(fs_representation, target_representation)
        else:
            return -1000


    algorithm = GA(search_space=original_problem.search_space,
                   crossover_operator=UniformFSCrossover(),
                   mutation_operator=SinglePointFSMutation(search_space=original_problem.search_space),
                   selection_operator=TournamentSelection(3),
                   crossover_rate=0.3,
                   elite_proportion=0.05,
                   population_size=population_size,
                   tournament_size=3,
                   fitness_function=fitness_function)

    algorithm.run(termination_criteria=FullSolutionEvaluationLimit(search_budget))

    winning_solution = algorithm.get_results(1)[0]
    return (winning_solution, representation.get_representation(winning_solution))