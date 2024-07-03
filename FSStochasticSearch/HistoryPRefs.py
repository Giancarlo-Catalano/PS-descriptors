from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper import TerminationCriteria
from FirstPaper.EvaluatedFS import EvaluatedFS
from FirstPaper.PRef import PRef
from FSStochasticSearch.GA import GA
from FSStochasticSearch.Operators import SinglePointFSMutation, TwoPointFSCrossover, TournamentSelection
from FSStochasticSearch.SA import SA


"""
This file contains many functions with which a Reference population could be generated.
"""

def uniformly_random_distribution_pRef(benchmark_problem: BenchmarkProblem,
                                       sample_size: int) -> PRef:
    """
    Res Ipsa Loquitur. The solutions are completely uniformly random
    @param benchmark_problem: which provides the search space that will be sampled
    @param sample_size: the amount of individuals that will be produced
    @return: the referece population, as a PRef
    """
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
    """
    Returns the reference population composed of all the attempts from some SA runs.
    Note that if the run ends and more individuals are needed, a new run will be used as well
    @param benchmark_problem: whose search space is sampled
    @param sample_size: the amount of individuals at the end
    @param max_trace: max non-improving attempts in the SA (unused generally)
    @return: the reference population, as a PRef
    """
    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))

    solutions : list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_trace(max_trace= max_trace))

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
    Note that this is significantly slower that using all of the attempts
    (This function is not used)"""

    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))

    solutions = [algorithm.get_one() for _ in range(sample_size)]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)
