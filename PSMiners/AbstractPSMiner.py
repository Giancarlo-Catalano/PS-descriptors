import heapq
import random
from math import ceil
from typing import Optional, TypeAlias

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.FSEvaluator import FSEvaluator
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from Core.TerminationCriteria import TerminationCriteria, PSEvaluationLimit
from utils import announce

Population: TypeAlias = list[EvaluatedPS]
ResultsAsJSON: TypeAlias = dict


class AbstractPSMiner:
    pRef: PRef

    def __init__(self,
                 pRef: PRef):
        self.pRef = pRef


    def __repr__(self):
        raise Exception("An implementation of AbstractPSMiner does not implement __repr__")

    @property
    def search_space(self):
        return self.pRef.search_space

    @staticmethod
    def get_mixed_initial_population(search_space: SearchSpace,
                                     from_uniform: float,
                                     from_half_fixed: float,
                                     from_geometric: float,
                                     population_size: int) -> Population:

        def uniform_random() -> PS:
            # note the rand(card + 1) - 1, which allows a number from the range [-1, card -1]
            return PS([random.randrange(cardinality + 1) - 1 for cardinality in search_space.cardinalities])

        def half_chance_random() -> PS:
            return PS([STAR if random.random() < 0.5
                       else random.randrange(cardinality)
                       for cardinality in search_space.cardinalities])

        def geometric_random_with_success_rate(success_chance: float) -> PS:
            total_var_count = search_space.amount_of_parameters

            def get_amount_of_fixed_vars():
                # The geometric distribution is simulated using bernoulli trials,
                # where each trial will add a fixed variable onto the ps
                result_count = 0
                while result_count < total_var_count:
                    if random.random() < success_chance:
                        result_count += 1
                    else:
                        break
                return result_count

            vars_to_include = random.choices(list(range(total_var_count)), k=get_amount_of_fixed_vars())

            result_values = [-1 for _ in range(total_var_count)]
            for included_var in vars_to_include:
                result_values[included_var] = random.randrange(search_space.cardinalities[included_var])
            return PS(result_values)

        def geometric_random():
            return geometric_random_with_success_rate(2 / 3)

        def generate_amount_with_function(generator, proportion):
            amount = ceil(proportion * population_size)
            return [generator() for _ in range(amount)]

        pss = []
        pss.extend(generate_amount_with_function(uniform_random, from_uniform))
        pss.extend(generate_amount_with_function(half_chance_random, from_half_fixed))
        pss.extend(generate_amount_with_function(geometric_random, from_geometric))

        return [EvaluatedPS(ps) for ps in pss]

    def step(self):
        raise Exception(f"An implementation of PSMiner ({self.__repr__()}) does not implement get_initial_population")

    def get_used_evaluations(self) -> int:
        raise Exception(f"An implementation of PSMiner ({self.__repr__()}) does not implement get_used_evaluations")

    def run(self, termination_criteria: TerminationCriteria):
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            ps_evaluations=self.get_used_evaluations())

        while not should_terminate():
            self.step()
            iterations +=1

    def get_results(self, amount: Optional[int]) -> list[EvaluatedPS]:
        raise Exception(f"An implementation of PSMiner({self.__repr__()}) does not implement get_results")

    @staticmethod
    def get_best_n(n: int, population: Population) -> Population:
        return heapq.nlargest(n=n, iterable=population)

    @staticmethod
    def without_duplicates(population: Population) -> Population:
        return list(set(population))

    def get_parameters_as_dict(self) -> dict:
        raise Exception(f"An implementation of PSMiner ({self.__repr__}) does not implement get_parameters_as_dict")



    @classmethod
    def with_default_settings(cls, pRef: PRef):
        raise Exception("An implementation of PSMiner does not implement .with_default_settings")


    @classmethod
    def test_with_problem(cls, benchmark_problem: BenchmarkProblem):
        evaluator = FSEvaluator(benchmark_problem.fitness_function)

        with announce("Gathering pRef"):
            pRef = evaluator.generate_pRef_from_search_space(search_space=benchmark_problem.search_space,
                                                             amount_of_samples=10000)
        with announce("Running the algorithm"):
            algorithm: AbstractPSMiner = cls.with_default_settings(pRef)
            algorithm.run(PSEvaluationLimit(15000))

        print("The best results are")
        best = algorithm.get_results(None)
        for item in best:
            print(item)