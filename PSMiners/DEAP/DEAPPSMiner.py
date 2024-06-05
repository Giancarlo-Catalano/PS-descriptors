from typing import Any, Optional

from deap.base import Toolbox
from deap.tools import Logbook

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.TerminationCriteria import TerminationCriteria, PSEvaluationLimit
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.DEAP.deap_utils import get_toolbox_for_problem, get_stats_object, nsga
from utils import announce


class DEAPPSMiner(AbstractPSMiner):
    population_size: int

    toolbox: Toolbox
    stats: Any
    classic3_evaluator: Classic3PSEvaluator
    uses_experimental_crowding: bool
    last_logbook: Optional[Logbook]
    last_population: Optional[list[EvaluatedPS]]

    def __init__(self,
                 pRef: PRef,
                 population_size: int,
                 uses_custom_crowding: bool,
                 use_spea = False):
        super().__init__(pRef=pRef)
        self.population_size = population_size
        self.uses_experimental_crowding = uses_custom_crowding

        self.classic3_evaluator = Classic3PSEvaluator(self.pRef)  # replaces simplicity, mean fitness, atomicity
        self.toolbox = get_toolbox_for_problem(pRef,
                                               classic3_evaluator=self.classic3_evaluator,
                                               uses_experimental_crowding=self.uses_experimental_crowding,
                                               use_spea=use_spea)
        self.stats = get_stats_object()

    def __repr__(self):
        return f"NSGAPSMiner(uses_experimental_crowding = {self.uses_experimental_crowding})"


    def get_used_evaluations(self) -> int:
        return self.classic3_evaluator.used_evaluations

    @classmethod
    def nsgaii_population_to_evaluated_ps_population(cls, nsga_population) -> list[EvaluatedPS]:
        def convert_single(nsga_individual):
            result = EvaluatedPS(nsga_individual)  # because nsgaindividual is a subclass of PS
            result.metric_scores = nsga_individual.fitness.values
            return result

        return [convert_single(individual) for individual in nsga_population]

    def run(self, termination_criteria: TerminationCriteria, verbose=False):
        final_population, self.last_logbook = nsga(toolbox=self.toolbox,
                                         mu =self.population_size,
                                         cxpb=0.5,
                                         mutpb=0.7,
                                         termination_criteria = termination_criteria,
                                         stats=self.stats,
                                         verbose=verbose,
                                         classic3_evaluator=self.classic3_evaluator)

        self.last_population = DEAPPSMiner.nsgaii_population_to_evaluated_ps_population(final_population)

    @classmethod
    def with_default_settings(cls, pRef: PRef):
        return cls(population_size = 300,
                   uses_custom_crowding = True,
                   pRef = pRef)




    def get_results(self, amount: Optional[int] = None) -> list[EvaluatedPS]:
        if amount is None:
            amount = len(self.last_population)

        self.last_population = utils.sort_by_combination_of(self.last_population, key_functions = [lambda x: x.metric_scores[0],
                                                                                          lambda x: x.metric_scores[1],
                                                                                          lambda x: x.metric_scores[2]], reverse=True)
        self.last_population = sorted(self.last_population, key=lambda x: -x.metric_scores[2])
        return self.last_population[:amount]



def test_DEAP_miner(benchmark_problem: BenchmarkProblem,
                    pRef: PRef,
                    budget: int,
                    custom_crowding: bool):
    miner = DEAPPSMiner(pRef, 150, custom_crowding, False)
    termination_criteria = PSEvaluationLimit(budget)


    with announce(f"Running NSGAIII in DEAP"):
        miner.run(termination_criteria, verbose=True)

    return miner.get_results()
