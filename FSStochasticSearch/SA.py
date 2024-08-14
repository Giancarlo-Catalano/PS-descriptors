import copy
import random
from typing import Callable

import numpy as np

from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution

from FSStochasticSearch.Operators import FSMutationOperator
from Core.SearchSpace import SearchSpace


def acceptance_probability(f_n, f_c, min_fitness, max_fitness, temperature) -> float:

    if max_fitness == min_fitness:
        return 1.0
    def normalise(f):
        return (f-min_fitness) / (max_fitness - min_fitness)

    return np.exp((normalise(f_n) - normalise(f_c)) / temperature)
class SA:
    cooling_coefficient: float
    search_space: SearchSpace

    mutation_operator: FSMutationOperator
    evaluator: FSEvaluator
    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 mutation_operator: FSMutationOperator,
                 cooling_coefficient = 0.99995):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)

        self.mutation_operator = mutation_operator
        self.cooling_coefficient = cooling_coefficient



    def get_one(self):
        current_individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
        current_individual.fitness = self.evaluator.evaluate(current_individual.full_solution)

        current_best = current_individual

        temperature = 1

        while temperature > 0.01:
            new_candidate_solution = self.mutation_operator.mutated(current_individual.full_solution)
            new_fitness = self.evaluator.evaluate(new_candidate_solution)
            new_candidate = EvaluatedFS(new_candidate_solution, new_fitness)

            passing_probability = acceptance_probability(new_candidate.fitness, current_individual.fitness, temperature)
            if new_candidate > current_individual or random.random() < passing_probability:
                current_individual = new_candidate
                if current_individual > current_best:
                    current_best = current_individual

            temperature *= self.cooling_coefficient

        return current_best


    def get_one_with_attempts(self, max_trace: int, consecutive_fail_termination = 100000) -> list[EvaluatedFS]:
        trace = []
        current_individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
        current_individual.fitness = self.evaluator.evaluate(current_individual)

        current_best = current_individual
        trace.append(copy.copy(current_individual))
        temperature = 1

        consecutive_fails = 0

        min_fitness = current_individual.fitness
        max_fitness = current_individual.fitness

        while temperature > 0.01 and consecutive_fails < consecutive_fail_termination and len(trace) < max_trace:

            new_candidate_solution = self.mutation_operator.mutated(current_individual)
            new_fitness = self.evaluator.evaluate(new_candidate_solution)
            new_candidate = EvaluatedFS(new_candidate_solution, new_fitness)

            if current_best >= current_individual:
                consecutive_fails += 1
            else:
                consecutive_fails = 0

            min_fitness = min(min_fitness, new_fitness)
            max_fitness = max(max_fitness, new_fitness)


            passing_probability = acceptance_probability(new_candidate.fitness,
                                                         current_individual.fitness,
                                                         min_fitness,
                                                         max_fitness,
                                                         temperature)
            #print(f"{consecutive_fails = }, {temperature:.2f}, {passing_probability:.3f}")
            if new_candidate > current_individual or random.random() < passing_probability:
                current_individual = new_candidate

                if current_individual > current_best:
                    current_best = current_individual

                trace.append(copy.copy(current_individual))



            temperature *= self.cooling_coefficient

        return trace


    def get_one_with_attempts_original(self, max_trace: int) -> list[EvaluatedFS]:
        trace = []
        current_individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
        current_individual.fitness = self.evaluator.evaluate(current_individual.full_solution)

        #current_best = current_individual
        trace.append(copy.copy(current_individual))
        temperature = 1


        while temperature > 0.01 and len(trace) < max_trace:
            new_candidate_solution = self.mutation_operator.mutated(current_individual.full_solution)
            new_fitness = self.evaluator.evaluate(new_candidate_solution)
            new_candidate = EvaluatedFS(new_candidate_solution, new_fitness)

            passing_probability = acceptance_probability(new_candidate.fitness, current_individual.fitness, temperature)
            if new_candidate > current_individual or random.random() < passing_probability:
                current_individual = new_candidate
                trace.append(copy.copy(current_individual))
                #if current_individual > current_best:
                #    current_best = current_individual



            temperature *= self.cooling_coefficient

        return trace






