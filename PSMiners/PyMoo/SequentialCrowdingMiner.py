from typing import Optional

import numpy as np
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize

import utils
from FirstPaper.EvaluatedPS import EvaluatedPS
from FirstPaper.PRef import PRef
from FirstPaper.PSMetric.Additivity import Influence
from FirstPaper.TerminationCriteria import TerminationCriteria, PSEvaluationLimit, UnionOfCriteria, IterationLimit, \
    SearchSpaceIsCovered
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.PyMoo.CustomCrowding import PyMooDecisionSpaceSequentialCrowding
from PSMiners.PyMoo.Operators import PSUniformMutation, PSGeometricSampling
from PSMiners.PyMoo.PSPyMooProblem import PSPyMooProblem
from PSMiners.PyMoo.pymoo_utilities import get_pymoo_search_algorithm
from utils import announce


class SequentialCrowdingMiner(AbstractPSMiner):
    """
    This is the algorithm which produces partial solutions for a given reference population
    """
    which_algorithm: str   # the MO algorithm to use
    population_size_per_run: int  # there are multiple MO runs, each with a population
    budget_per_run: int  # and a budget

    pymoo_problem: PSPyMooProblem  # the pymoo problem instance, the task of finding PSs
    winners_archive: list[EvaluatedPS] # all the PSs returned by the runs

    use_custom_crowding_operator: bool
    influence_metric: Influence   # used to better sort PSs, for aesthetic reasons


    def __init__(self,
                 pRef: PRef,
                 which_algorithm: str,
                 population_size_per_run: int,
                 budget_per_run: int,
                 use_experimental_crowding_operator: bool = True,
                 influence_metric: Optional[Influence] = None):
        super().__init__(pRef=pRef)
        self.which_algorithm = which_algorithm
        self.population_size_per_run = population_size_per_run
        self.budget_per_run = budget_per_run
        self.pymoo_problem = PSPyMooProblem(pRef)
        self.winners_archive = []
        self.use_custom_crowding_operator = use_experimental_crowding_operator

        if influence_metric is None:
            influence_metric = Influence()
            influence_metric.set_pRef(self.pRef)
        self.influence_metric = influence_metric


    def __repr__(self):
        return (f"SequentialCrowdingMiner({self.which_algorithm = }, "
                f"{self.population_size_per_run =}, "
                f"{self.budget_per_run})")


    def get_used_evaluations(self) -> int:
        return self.pymoo_problem.objectives_evaluator.used_evaluations

    @classmethod
    def output_of_miner_to_evaluated_ps(cls, output_of_miner) -> list[EvaluatedPS]:
            return [EvaluatedPS(values, metric_scores=ms)
                     for values, ms in zip(output_of_miner.X, output_of_miner.F)]


    @classmethod
    def sort_by_atomicity(cls, e_pss: list[EvaluatedPS]) -> list[EvaluatedPS]:
        e_pss.sort(reverse=False, key=lambda x: x.metric_scores[-1])   # reverse is false because the keys are inverted
        return e_pss

    @classmethod
    def sort_by_mean_fitness(cls, e_pss: list[EvaluatedPS]) -> list[EvaluatedPS]:
        e_pss.sort(reverse=False, key=lambda x: x.metric_scores[1])
        return e_pss



    @classmethod
    def sort_by_mean_fitness_and_atomicity(cls, e_pss: list[EvaluatedPS]):
        return utils.sort_by_combination_of(e_pss,
                                            key_functions=[lambda x: x.metric_scores[1],
                                                           lambda x: x.metric_scores[2]],
                                            reverse=False)



    def get_crowding_operator(self):
        if self.use_custom_crowding_operator and len(self.winners_archive) > 0:
            # this is the crowding operator discussed in the paper
            return PyMooDecisionSpaceSequentialCrowding(archived_pss=self.winners_archive, sigma_shared=0.5)
        else:
            # on the first run, we are forced to run the default crowding operator
            return RankAndCrowding()

    def get_miner_algorithm(self):
        return get_pymoo_search_algorithm(which_algorithm=self.which_algorithm,
                                          pop_size=self.population_size_per_run,
                                          sampling=PSGeometricSampling(),
                                          mutation=PSUniformMutation(self.search_space),
                                          crossover=UniformCrossover(prob=0),  # as the first paper discussed, crossover is bad
                                          crowding_operator=self.get_crowding_operator(),
                                          search_space=self.search_space)



    def get_coverage(self):
        """
        This function is for reporting purposes only, and tells how many variables have appeared in PSs so far
        @return: an array, with the proportions of how many times a variable appears in the archive
        """
        if len(self.winners_archive) == 0:
            return np.zeros(self.search_space.amount_of_parameters)
        else:
            return PyMooDecisionSpaceSequentialCrowding.get_coverage(search_space=self.pymoo_problem.search_space,
                                                          already_obtained=self.winners_archive)


    def sort_by_clarity(self, pss: list[EvaluatedPS]) -> list[EvaluatedPS]:
        def get_influence_delta(ps: EvaluatedPS) -> float:
            return self.influence_metric.get_single_score(ps)

        return utils.sort_by_combination_of(pss, key_functions=[get_influence_delta], reverse=True)

    def step(self, verbose = False):
        algorithm = self.get_miner_algorithm()
        if verbose:
            coverage = self.get_coverage()
            print(f"In the operator, the coverage is {(coverage*100).astype(int)}")

        with announce("Running a single search step", verbose):
            res = minimize(self.pymoo_problem,
                           algorithm,
                           termination=('n_evals', self.budget_per_run),
                           verbose=verbose)


        winners = self.output_of_miner_to_evaluated_ps(res)
        self.winners_archive.extend(winners)


        if verbose:
            print("At the end of this run, the winners were")
            for winner in winners:
                print(winner)


    def run(self, termination_criteria: TerminationCriteria, verbose=False):
        iterations = 0
        def should_stop():
            return termination_criteria.met(ps_evaluations = self.get_used_evaluations(),
                                            archive = self.winners_archive,
                                            coverage = self.get_coverage(),
                                            iterations = iterations)


        while not should_stop():
            if verbose:
                print(f"Evaluations: {self.get_used_evaluations()}")
            self.step(verbose=verbose)
            iterations += 1

    @classmethod
    def with_default_settings(cls, pRef: PRef):
        """ These are the settings that were found via the testing in the paper"""
        return cls(pRef = pRef,
                   budget_per_run = 1000,
                   population_size_per_run = 50,
                   which_algorithm="NSGAII")



    def get_results(self, amount: Optional[int] = None) -> list[EvaluatedPS]:
        if amount is None:
            amount = len(self.winners_archive)

        self.winners_archive = self.without_duplicates(self.winners_archive)
        self.winners_archive = self.sort_by_atomicity(self.winners_archive)
        return self.winners_archive[:amount]

