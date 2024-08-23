import heapq
import random
from math import floor
from typing import Callable, TypeAlias

from Core import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator, FSCrossoverOperator, FSSelectionOperator, TournamentSelection, \
    SinglePointFSMutation, TwoPointFSCrossover

Population: TypeAlias = list[EvaluatedFS]

class GA:
    search_space: SearchSpace
    mutation_operator: FSMutationOperator
    crossover_operator: FSCrossoverOperator
    selection_operator: TournamentSelection
    crossover_rate: float

    elite_proportion: float
    tournament_size: int
    population_size: int
    termination_criteria: TerminationCriteria

    evaluator: FSEvaluator

    current_population: Population

    def __init__(self,
                 search_space: SearchSpace,
                 mutation_operator: FSMutationOperator,
                 crossover_operator: FSCrossoverOperator,
                 selection_operator: FSSelectionOperator,
                 crossover_rate: float,
                 elite_proportion: float,
                 tournament_size: int,
                 population_size: int,
                 fitness_function: Callable[[FullSolution], float],
                 starting_population=None):
        self.search_space = search_space
        self.mutation_operator = mutation_operator
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.crossover_rate = crossover_rate
        self.elite_proportion = elite_proportion
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.evaluator = FSEvaluator(fitness_function)

        if starting_population is None:
            self.current_population = self.get_initial_population()
        else:
            self.current_population = starting_population

        self.current_population = self.evaluator.evaluate_population(self.current_population)

    def random_solution(self) -> FullSolution:
        return FullSolution.random(self.search_space)

    def get_initial_population(self) -> Population:
        return [EvaluatedFS(self.random_solution(), 0) for _ in range(self.population_size)]

    def get_elite(self) -> list[EvaluatedFS]:
        amount = floor(self.elite_proportion * len(self.current_population))
        return heapq.nlargest(amount, self.current_population)


    def select_one(self) -> EvaluatedFS:
        return self.selection_operator.select_single(population=self.current_population)

    def make_new_child(self) -> EvaluatedFS:
        if random.random() < self.crossover_rate:
            # do crossover
            mother = self.select_one()
            father = self.select_one()

            child_ps = self.mutation_operator.mutated(self.crossover_operator.crossed(mother, father))
        else:
            child_ps = self.mutation_operator.mutated(self.select_one())
        return EvaluatedFS(child_ps, 0)

    def make_new_evaluated_population(self) -> list[EvaluatedFS]:
        elite = self.get_elite()
        children = [self.make_new_child()
                    for _ in range(self.population_size - len(elite))]
        children = self.evaluator.evaluate_population(children)
        return elite + children

    def step(self):
        self.current_population = self.make_new_evaluated_population()

    def run(self,
            termination_criteria: TerminationCriteria.TerminationCriteria,
            show_every_generation=False):
        iteration = 0

        def termination_criteria_met():
            return termination_criteria.met(iterations=iteration,
                                            fs_evaluations=self.evaluator.used_evaluations,
                                            evaluated_population=self.current_population)

        while not termination_criteria_met():
            if show_every_generation:
                self.show_current_state()
            self.step()
            iteration += 1

    def get_current_best(self) -> EvaluatedFS:
        return max(self.current_population)

    def show_current_state(self):
        print(f"The current best fitness is {self.get_current_best()}.")


    def get_results(self, quantity_returned: int) -> list[EvaluatedFS]:
        return heapq.nlargest(quantity_returned, self.current_population)


def test_FSGA(benchmark_problem: BenchmarkProblem):
    print("Testing the full solution FSStochasticSearch")
    algorithm = GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=500,
                   fitness_function=benchmark_problem.fitness_function)

    print("Now running the algorithm")
    termination_criterion = TerminationCriteria.FullSolutionEvaluationLimit(20000)
    algorithm.run(termination_criterion, show_every_generation=True)

    print("The algorithm has terminated, and the results are")
    results = algorithm.get_results(12)

    for individual in results:
        print(f"{individual}")