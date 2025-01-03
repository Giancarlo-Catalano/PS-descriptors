from collections import deque
from typing import Callable

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator, SinglePointFSMutation


class TabuSearch:
    tabu_list_size: int
    gradient_samples: int

    mutation_operator: FSMutationOperator

    evaluator: FSEvaluator

    def __init__(self,
                 fitness_function: Callable,
                 mutation_operator: FSMutationOperator,
                 tabu_list_size: int = 50,
                 gradient_samples: int = 200):
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator

        self.tabu_list_size = tabu_list_size
        self.gradient_samples = gradient_samples

    def get_random_individual(self) -> FullSolution:
        return FullSolution.random(self.mutation_operator.search_space)

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:

        def tweak_copy(solution):
            return self.mutation_operator.mutated(solution)

        def evaluated(solution):
            return EvaluatedFS(solution, self.evaluator.evaluate(solution))

        # this is based on page 26 of Essentials of Metaheuristics

        trace: list[EvaluatedFS] = []

        s = evaluated(self.get_random_individual())
        #best = s
        tabu_list = deque()
        tabu_set = set()   # it's faster to check if something is here...

        def add_to_tabu_list(solution):
            tabu_list.append(s)
            tabu_set.add(s)
            trace.append(s)

        def remove_oldest_item_from_tabu_list():
            removed = tabu_list.popleft()
            tabu_set.discard(removed)

        add_to_tabu_list(s)

        while len(trace) < max_trace:
            if len(tabu_list) > self.tabu_list_size:
                remove_oldest_item_from_tabu_list()

            r = evaluated(tweak_copy(s))

            for _ in range(self.gradient_samples):
                w = self.mutation_operator.mutated(s)
                if w in tabu_set:
                    continue
                w = evaluated(w)
                r_in_tabu_list = r in tabu_set
                if w > r or r_in_tabu_list:
                    r = w

                if not r_in_tabu_list:
                    s = r
                    add_to_tabu_list(r)

                # we don't need to update best...
        return trace




