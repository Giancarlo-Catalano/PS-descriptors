import json
import re
from typing import Iterable

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.InverseGraphColouringProblem import InverseGraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core import TerminationCriteria
from Core.PRef import plot_solutions_in_pRef
from Core.PS import PS, STAR
from Core.PSMetric.Linkage.Additivity import Influence
from Explanation.PRefManager import PRefManager
from PSMiners.PyMoo.SequentialCrowdingMiner import SequentialCrowdingMiner
import logging

from utils import execution_timer

logger = logging.getLogger(__name__)



def get_error_between_pss(ps_a: PS, ps_b: PS) -> int:
    return int(np.sum(ps_a.values != ps_b.values))


def minimum_error_to_find(target: PS, pool: list[PS]) -> int:
    return min(get_error_between_pss(target, ps) for ps in pool)


class HyperparameterEvaluator:
    pRef_sizes_to_test: list[int]
    pRef_origin_methods: list[str]
    problems_to_test: list[str]
    algorithms_to_test: list[str]
    population_sizes_to_test: list[int]
    ps_budget: int
    ps_budgets_per_run_to_test: list[int]
    custom_crowding_operators_to_test: list[bool]


    def __init__(self,
                 pRef_sizes_to_test: list[int],
                 pRef_origin_methods: list[str],
                 problems_to_test: list[str],
                 algorithms_to_test: list[str],
                 population_sizes_to_test: list[int],
                 ps_budgets_per_run_to_test: list[int],
                 ps_budget: int,
                 custom_crowding_operators_to_test: list[bool]):
        self.pRef_sizes_to_test = pRef_sizes_to_test
        self.pRef_origin_methods = pRef_origin_methods
        self.problems_to_test = problems_to_test
        self.algorithms_to_test = algorithms_to_test
        self.population_sizes_to_test = population_sizes_to_test
        self.ps_budgets_per_run_to_test = ps_budgets_per_run_to_test
        self.ps_budget = ps_budget
        self.custom_crowding_operators_to_test = custom_crowding_operators_to_test



    def get_problem_from_str(self, problem_str: str) -> (BenchmarkProblem, BenchmarkProblem):
        parameters = re.findall(r"\d+", problem_str)
        parameters = [int(v) for v in parameters]
        if problem_str.startswith("insular") or problem_str.startswith("island"):
            gc_problem = GraphColouring.make_insular_instance(amount_of_islands=parameters[0])
            bt_problem = EfficientBTProblem.from_Graph_Colouring(gc_problem)
            return gc_problem, bt_problem
        elif problem_str.startswith("RR"):
            if len(parameters) < 2:
                clique_size = 4
            else:
                clique_size = parameters[1]
            rr_problem = RoyalRoad(amount_of_cliques=parameters[0], clique_size=clique_size)
            bt_problem = EfficientBTProblem.from_RoyalRoad(rr_problem)
            return rr_problem, bt_problem
        elif problem_str.startswith("collaboration"):
            clique_size = 4 if len(parameters) < 2 else parameters[1]
            colour_count = 2 if len(parameters) < 3 else parameters[2]
            igc_problem = InverseGraphColouring(amount_of_cliques=parameters[0],
                                                clique_size=clique_size,
                                                amount_of_colours=colour_count)
            bt_problem = EfficientBTProblem.from_inverse_graph_colouring_problem(igc_problem)
            return igc_problem, bt_problem
        else:
            raise Exception("The problem string was not recognised")

    @classmethod
    def get_count_of_covered_vars(cls, mined_pss: Iterable[PS]) -> int:
        covered_positions = np.array([ps.values != STAR for ps in mined_pss])
        covered_positions = np.sum(covered_positions, axis=0) > 0
        return int(np.sum(covered_positions))


    def get_data(self, ignore_errors = False, verbose=False):
        results = []
        for problem_str in self.problems_to_test:
            original_problem, bt_problem = self.get_problem_from_str(problem_str)
            targets: set[PS] = original_problem.get_targets()
            for pRef_origin_method in self.pRef_origin_methods:
                for pRef_size in self.pRef_sizes_to_test:
                    logger.info(f"Generating the pRef")
                    pRef = PRefManager.generate_pRef(problem = bt_problem,
                                                     which_algorithm= pRef_origin_method,
                                                     sample_size=pRef_size)
                    influence_evaluator = Influence()
                    influence_evaluator.set_pRef(pRef)
                    if verbose:
                        plot_solutions_in_pRef(pRef, "dummy_name.png")
                    for miner_algorithm in self.algorithms_to_test:
                        for uses_custom_crowding_operator in self.custom_crowding_operators_to_test:
                            for population_size in self.population_sizes_to_test:
                                for ps_budget_per_run in self.ps_budgets_per_run_to_test:
                                    logger.info(f"{problem_str = }, "
                                          f"{pRef_origin_method = },"
                                          f"{pRef_size = }, "
                                          f"{miner_algorithm = }, "
                                          f"{population_size = }, "
                                          f"{ps_budget_per_run = },"
                                          f"{uses_custom_crowding_operator = },"
                                          f"ps_budget = {self.ps_budget}")
                                    try:
                                        miner = SequentialCrowdingMiner(pRef = pRef,
                                                                        which_algorithm=miner_algorithm,
                                                                        population_size_per_run=population_size,
                                                                        budget_per_run=ps_budget_per_run,
                                                                        use_experimental_crowding_operator=uses_custom_crowding_operator,
                                                                        influence_metric=influence_evaluator)
                                        eval_limit = TerminationCriteria.PSEvaluationLimit(self.ps_budget)
                                        finish_early_if_all_found = TerminationCriteria.UntilAllTargetsFound(targets)
                                        termination_criteria = TerminationCriteria.UnionOfCriteria(eval_limit, finish_early_if_all_found)
                                        with execution_timer() as timer:
                                            miner.run(termination_criteria, verbose=verbose)
                                        mined_pss = miner.get_results()
                                        found = targets.intersection(mined_pss)
                                        smallest_errors = [minimum_error_to_find(target, mined_pss) for target in targets]
                                        logger.info(f"\t{len(found)} were found!")
                                        datapoint = {"total_ps_budget": self.ps_budget,
                                                     "problem_str": problem_str,
                                                     "pRef_origin_method": pRef_origin_method,
                                                     "pRef_size": pRef_size,
                                                     "miner_algorithm": miner_algorithm,
                                                     "population_size": population_size,
                                                     "ps_budget_per_run": ps_budget_per_run,
                                                     "mined_count": len(mined_pss),
                                                     "target_count": len(targets),
                                                     "found_count": len(found),
                                                     "smallest_errors": [smallest_errors],
                                                     "uses_custom_crowding_operator": uses_custom_crowding_operator,
                                                     "vars_covered": self.get_count_of_covered_vars(found),
                                                     "runtime":timer.execution_timer,
                                                     "used_evaluations":miner.get_used_evaluations()}
                                    except Exception as e:  # TODO change this
                                        if not ignore_errors:
                                            raise e
                                        logger.info(f"An error {e} occurred")
                                        datapoint = {"ERROR": repr(e),
                                                     "total_ps_budget": self.ps_budget,
                                                     "problem_str": problem_str,
                                                     "pRef_origin_method": pRef_origin_method,
                                                     "miner_algorithm": miner_algorithm,
                                                     "population_size": population_size,
                                                     "ps_budget_per_run": ps_budget_per_run,
                                                     "uses_custom_crowding_operator": uses_custom_crowding_operator
                                                     }
                                    results.append(datapoint)
        print(json.dumps(results, indent=4))

