from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from FSStochasticSearch.GA import GA
from FSStochasticSearch.Operators import SinglePointFSMutation, TwoPointFSCrossover, TournamentSelection
from FSStochasticSearch.SA import SA
from FSStochasticSearch.TabuSearch import TabuSearch


def uniformly_random_distribution_pRef(benchmark_problem: BenchmarkProblem,
                                       sample_size: int) -> PRef:
    return benchmark_problem.get_reference_population(sample_size=sample_size)


def pRef_from_GA(benchmark_problem: BenchmarkProblem,
                 ga_population_size: int,
                 sample_size: int) -> PRef:
    """returns the population obtained by concatenating all the generations the GA will go through"""
    algorithm = GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=ga_population_size,
                   fitness_function=benchmark_problem.fitness_function)

    solutions : list[EvaluatedFS] = []
    solutions.extend(algorithm.current_population)

    while len(solutions) < sample_size:
        algorithm.step()
        solutions.extend(algorithm.current_population)

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA(benchmark_problem: BenchmarkProblem,
                 sample_size: int,
                 max_trace: int) -> PRef:
    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions : list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace= max_trace))

    solutions = solutions[:sample_size]

    # best_solution = max(solutions)
    # df = benchmark_problem.details_of_solution(best_solution.full_solution)   # Experimental
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_tabu_search(benchmark_problem: BenchmarkProblem,
                          sample_size: int,
                          max_trace: int) -> PRef:
    algorithm = TabuSearch(fitness_function=benchmark_problem.fitness_function,
                           mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))

    solutions: list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))

    solutions = solutions[:sample_size]

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)




def pRef_from_GA_best(benchmark_problem: BenchmarkProblem,
                      fs_evaluation_budget: int,
                      sample_size: int) -> PRef:
    """
    Returns the population of the last iteration of the algorithm, after having used the given evaluation budget.
    """
    algorithm =  GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=sample_size,
                   fitness_function=benchmark_problem.fitness_function)


    algorithm.run(termination_criteria=TerminationCriteria.FullSolutionEvaluationLimit(fs_evaluation_budget))
    solutions = algorithm.get_results(sample_size)

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA_best(benchmark_problem: BenchmarkProblem,
                 sample_size: int) -> PRef:
    """returns only the end results of each run of SA. There will be _sample\_size_ runs in total.
    Note that this is significantly slower that using all of the attempts"""

    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions = [algorithm.get_one() for _ in range(sample_size)]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

