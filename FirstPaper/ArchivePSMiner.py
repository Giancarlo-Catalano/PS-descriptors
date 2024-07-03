import heapq
import json
import warnings
from typing import Optional, TypeAlias

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.EvaluatedPS import EvaluatedPS
from FirstPaper.PRef import PRef
from FirstPaper.PS import PS
from FirstPaper.PSMetric.LegacyAtomicity import Atomicity
from FirstPaper.PSMetric.MeanFitness import MeanFitness
from FirstPaper.PSMetric.Metric import Metric
from FirstPaper.PSMetric.Simplicity import Simplicity
from FirstPaper.SearchSpace import SearchSpace
from FirstPaper.TerminationCriteria import TerminationCriteria, PSEvaluationLimit, IterationLimit
from FirstPaper.get_init import just_empty
from FirstPaper.get_local import specialisations
from FirstPaper.selection import truncation_selection
from FSStochasticSearch.GA import GA
from FSStochasticSearch.Operators import SinglePointFSMutation, TwoPointFSCrossover, TournamentSelection
from PSMiners.AbstractPSMiner import AbstractPSMiner

Population: TypeAlias = list[EvaluatedPS]
GetInitType: TypeAlias = [[PRef, Optional[int]], list[PS]]
GetLocalType: TypeAlias = [[PS, SearchSpace], list[PS]]
SelectionType: TypeAlias = [[list[EvaluatedPS], int], list[EvaluatedPS]]


class ArchivePSMiner(AbstractPSMiner):
    """This class is the FirstPaper miner, which outputs a FirstPaper catalog when used right"""
    """There are many parts that can be modified, and these were tested in the paper, 
    but you should probably just use with_default_settings as a constructor"""

    metrics: list[Metric]  # usually they are simplicity, mean_fitness, atomicity
    population_size: int
    get_init: GetInitType  # generates the initial population
    get_local: GetLocalType  # generates the offspring of a selected ps

    selection: SelectionType  # the selection operator

    current_population: list[EvaluatedPS]
    archive: set[EvaluatedPS]  # the archive, which will contain all the selected PSs

    used_evaluations: int  # counts how many F_\psi evaluations have happened

    def __init__(self,
                 pRef: PRef,
                 metrics: list[Metric],
                 get_init: GetInitType,
                 get_local: GetLocalType,
                 population_size: int,
                 selection: SelectionType):
        super().__init__(pRef)
        self.used_evaluations = 0

        self.pRef = pRef
        self.metrics = metrics

        for metric in self.metrics:
            metric.set_pRef(self.pRef)

        self.get_init = get_init
        self.get_local = get_local
        self.selection = selection
        self.population_size = population_size

        self.current_population = [EvaluatedPS(ps) for ps in self.get_init(self.pRef, quantity=self.population_size)]
        self.current_population = self.evaluate_individuals(self.current_population)  # experimental
        self.archive = set()

    def __repr__(self):
        return f"PSMiner(population_size = {self.population_size})"

    @property
    def search_space(self):
        return self.pRef.search_space

    def with_aggregated_scores(self, population: list[EvaluatedPS]) -> list[EvaluatedPS]:
        """
        This is kinda the fitness function of PSs, where we
         - remap every metric between individuals, to be in range[0, 1]
         - average the metrics within individuals, to have a single value

         Note that the final fitnesses are RELATIVE to the population, which is why the algorithm is quite slow
        :param population: the population, where ALL of the metrics are assumed to have been calculated
        :return: population: the same population, but now .aggregated_score is a valid value
        """
        metric_matrix = np.array([ind.metric_scores for ind in population])
        for column in range(metric_matrix.shape[1]):
            if not isinstance(self.metrics[column], MeanFitness):
                metric_matrix[:, column] = utils.remap_array_in_zero_one(metric_matrix[:, column])

        averages = np.average(metric_matrix, axis=1)
        for individual, score in zip(population, averages):
            individual.aggregated_score = score

        return population

    def step(self):
        """ The contents of the main loop"""

        self.current_population = self.without_duplicates(self.current_population)

        # aggregate the various objectives into a single score
        self.current_population = self.with_aggregated_scores(self.current_population)
        # truncate population
        self.current_population = self.top(n=self.population_size, population=self.current_population)

        # select parents
        parents = self.selection(self.current_population, self.population_size // 3)
        parents = self.without_duplicates(parents)

        # get offspring
        children = [EvaluatedPS(child) for parent in parents for child in parent.specialisations(self.search_space)]

        # add selected individuals to archive
        self.archive.update(parents)

        self.current_population.extend(children)

        # remove from population the individuals that appear in the archive (including the parents]
        self.current_population = [ind for ind in self.current_population if ind not in self.archive]

        self.current_population = self.evaluate_individuals(self.current_population)

    def evaluate_individuals(self, newborns: Population) -> Population:
        """
        Calculates the metrics for each individual, but this is not the true fitness function!
        These metrics are ABSOLUTE, ie they are not relative to the population, although they are relative to the PRef.
        :param newborns: the individuals to be evaluated
        :return: the same individuals as the input, but now .metric_scores will be valid
        """
        for individual in newborns:
            if individual.metric_scores is None:  # avoid recalculating if already valid
                individual.metric_scores = [metric.get_single_score(individual) for metric in self.metrics]
                self.used_evaluations += 1
        return newborns

    def get_used_evaluations(self) -> int:
        return self.used_evaluations

    def run(self, termination_criteria: TerminationCriteria, verbose=False):
        """ Executes the main loop, with the termination criterion usually being an evaluation budget"""
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            ps_evaluations=self.get_used_evaluations()) or len(self.current_population) == 0

        while not should_terminate():
            self.step()
            if verbose:
                print(f"Current used budget is {self.used_evaluations}")
            iterations += 1

    def get_results(self, amount: Optional[int]) -> list[EvaluatedPS]:
        """
        This is the only way you should get the result out of this!!
        This method will evaluate the archive and return the top n
        Note that the evaluation is relative to the archive, so the aggregated scores are recalculated
        :param amount:
        :return: The best PSs in the archive, of the quantity specified
        """
        if amount is None:
            amount = len(self.archive)
        evaluated_archive = self.with_aggregated_scores(list(self.archive))
        return self.top(n=amount, population=evaluated_archive)

    @staticmethod
    def top(n: int, population: Population) -> Population:
        """ Same as in the paper, very straightforward"""
        return heapq.nlargest(n=n, iterable=population)

    @staticmethod
    def without_duplicates(population: Population) -> Population:
        return list(set(population))

    @classmethod
    def with_default_settings(cls, pRef: PRef):
        """ atomicity can be measured in many many ways, and the paper suggest an approach that I've improved over time"""
        """The function defined in the paper uses Atomicity(), but you should also try:
            - Linkage(): faster
            - BivariateLocalPerturbation(): much more accurate, but sloooow
            - BivariateANOVALinkage(): slow but more mathematically sound
            
        """
        return cls(population_size=300,
                   pRef=pRef,
                   metrics=[Simplicity(), MeanFitness(), Atomicity()],
                   get_init=just_empty,
                   get_local=specialisations,
                   selection=truncation_selection)







def measure_T2_success_rate(benchmark_problem:BenchmarkProblem):
    pRef_size = 10**4
    generations_to_evolve_for = list(range(0, 5, 5))
    budget = 10**5

    ga = GA(search_space=benchmark_problem.search_space,
            mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
            crossover_operator=TwoPointFSCrossover(),
            selection_operator=TournamentSelection(),
            crossover_rate=0.5,
            elite_proportion=3,
            tournament_size=3,
            population_size=pRef_size,
            fitness_function=benchmark_problem.fitness_function)

    total_generations = 0

    def run_generations_and_get_pRef(next_generation: int) -> PRef:
        generations_to_execute = next_generation - total_generations
        termination_criterion = IterationLimit(generations_to_execute)
        ga.run(termination_criterion)
        population = ga.current_population
        solutions, fitnesses = utils.unzip([(ind.full_solution, ind.fitness) for ind in population])
        return PRef.from_full_solutions(full_solutions=solutions,
                                        fitness_values=fitnesses,
                                        search_space=benchmark_problem.search_space)


    targets = benchmark_problem.get_targets()
    targets = [EvaluatedPS(ps) for ps in targets]   # makes line (1) easier to implement
    def run_ps_miner(pRef: PRef) -> int:
        ps_miner = ArchivePSMiner.with_default_settings(pRef)
        ps_miner.run(termination_criteria=PSEvaluationLimit(budget))
        results = ps_miner.get_results(50)
        return len([target for target in targets if target in results])  # (1)

    result_dict = {}
    for generation in generations_to_evolve_for:
        with utils.execution_time() as pref_timer:
            new_pRef = run_generations_and_get_pRef(generation)
            total_generations = generation
        warnings.warn(f"Generating the pref for generation {generation} took {pref_timer.execution_time} seconds")
        with utils.execution_time() as miner_timer:
            found_targets: int = run_ps_miner(new_pRef)
        warnings.warn(f"The run to {generation} took {miner_timer.execution_time} seconds")
        result_dict[generation] = found_targets

    print(json.dumps(result_dict))
