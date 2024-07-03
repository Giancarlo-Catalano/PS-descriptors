import copy
import random
from typing import Callable

import numpy as np

from FirstPaper.EvaluatedFS import EvaluatedFS
from FirstPaper.FSEvaluator import FSEvaluator
from FirstPaper.FullSolution import FullSolution

from FSStochasticSearch.Operators import FSMutationOperator
from FirstPaper.SearchSpace import SearchSpace

"""
This file contains the Simulated Annealing algorithm described by our industrial partner, in the paper
 Workforce rostering via metaheuristics,
by Mary Dimitropoulaki, Mathias Kern, Gilbert Owusu and Alistair McCormick


It is fairly simple, and the only modification is that we return all the generated candidate solutions 
rather than just the last one"""



def acceptance_probability(f_n: float, f_c: float, min_fitness: float, max_fitness: float, temperature: float) -> float:
    """
    @param f_n: the fitness of the new individual
    @param f_c: the fitness of the currently held individual
    @param min_fitness: the minimum observed fitness (for normalisation)
    @param max_fitness: the maximum observed fitness (for normalisation)
    @param temperature: the current temperature
    @return: The probability of having the new individual replace the current (when f_n < f_c)
    """
    if max_fitness == min_fitness:
        return 1.0

    def normalise(f):
        return (f - min_fitness) / (max_fitness - min_fitness)

    return np.exp((normalise(f_n) - normalise(f_c)) / temperature)


class SA:
    cooling_coefficient: float   # the multiplier for the temperature at each iteration
    search_space: SearchSpace  # the search space of the solutions

    mutation_operator: FSMutationOperator
    evaluator: FSEvaluator  # used to evaluate the solutions and keep track of the used evaluation count

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 mutation_operator: FSMutationOperator,
                 cooling_coefficient=0.9995):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)

        self.mutation_operator = mutation_operator
        self.cooling_coefficient = cooling_coefficient

    def get_one(self) -> EvaluatedFS:
        """
        This is the normal implementation of the function, which was expanded in the get_one_with_trace function
        Note that this function is outdated because it does not normalise the fitness!
        @return: a single individual, as a EvaluatedFS (full solution)
        """

        # create the initial individual
        current_individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
        current_individual.fitness = self.evaluator.evaluate(current_individual.full_solution)

        current_best = current_individual

        temperature = 1

        while temperature > 0.01:
            # create a new individual via mutation, and evaluate it
            new_candidate_solution = self.mutation_operator.mutated(current_individual.full_solution)
            new_fitness = self.evaluator.evaluate(new_candidate_solution)
            new_candidate = EvaluatedFS(new_candidate_solution, new_fitness)

            # accept always when fitness is better, accept with a certain probability otherwise
            accept = False
            if new_candidate > current_individual:
                accept = True
            else:
                passing_probability = acceptance_probability(new_candidate.fitness,
                                                             current_individual.fitness,
                                                             min_fitness = None,  # this is invalid
                                                             max_fitness= None,
                                                             temperature=temperature)
                accept = random.random() < passing_probability

            if accept: # replace the current individual and update the current best
                current_individual = new_candidate
                if current_individual > current_best:
                    current_best = current_individual

            # lower the temperature
            temperature *= self.cooling_coefficient

        return current_best

    def get_one_with_trace(self, max_trace: int, consecutive_fail_termination=100000) -> list[EvaluatedFS]:
        """
        Similar to the function above, but it returned all of the candidate solutions generated
        @param max_trace: the maximum size allowed for the trace. When this is exceeded, the run is terminated
        @param consecutive_fail_termination: the maximum amount of non-improving moves allowed before termination
        @return: a list of all the created candidate solutions, in order (the last is the best)
        """

        trace = []
        current_individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
        current_individual.fitness = self.evaluator.evaluate(current_individual.full_solution)

        current_best = current_individual
        trace.append(copy.copy(current_individual))
        temperature = 1

        consecutive_fails = 0

        # these are required in order to normalise the fitness to calculate the acceptance probability
        min_fitness = current_individual.fitness
        max_fitness = current_individual.fitness

        while temperature > 0.01 and consecutive_fails < consecutive_fail_termination and len(trace) < max_trace:
            # generate a new solution through mutation
            new_candidate_solution = self.mutation_operator.mutated(current_individual.full_solution)
            new_fitness = self.evaluator.evaluate(new_candidate_solution)
            new_candidate = EvaluatedFS(new_candidate_solution, new_fitness)

            # check if the iteration will do anything good
            if current_best >= current_individual:
                consecutive_fails += 1
            else:
                consecutive_fails = 0

            # update the fitness range
            min_fitness = min(min_fitness, new_fitness)
            max_fitness = max(max_fitness, new_fitness)

            # accept always when fitness is better, accept with a certain probability otherwise
            accept = False
            if new_candidate > current_individual:
                accept = True
            else:
                passing_probability = acceptance_probability(new_candidate.fitness,
                                                             current_individual.fitness,
                                                             min_fitness,
                                                             max_fitness,
                                                             temperature)
                accept = random.random() < passing_probability


            if accept: # replace the current individual and update the current best
                current_individual = new_candidate

                if current_individual > current_best:
                    current_best = current_individual

                # add the accepted individual to the trace
                trace.append(copy.copy(current_individual))

            # reduce temperature
            temperature *= self.cooling_coefficient

        return trace
