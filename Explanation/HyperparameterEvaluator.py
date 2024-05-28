import json
from typing import Iterator

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from Core import TerminationCriteria
from Core.PRef import PRef
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from PSMiners.PyMoo.SequentialCrowdingMiner import SequentialCrowdingMiner
import logging

from utils import execution_time

logger = logging.getLogger(__name__)


class HyperparameterEvaluator:
    pRef_sizes_to_test: list[int]
    pRef_origin_methods: list[str]
    island_amounts_to_test: list[int]
    algorithms_to_test: list[str]
    population_sizes_to_test: list[int]
    ps_budget: int
    ps_budgets_per_run_to_test: list[int]
    custom_crowding_operators_to_test: list[bool]


    def __init__(self,
                 pRef_sizes_to_test: list[int],
                 pRef_origin_methods: list[str],
                 island_amounts_to_test: list[int],
                 algorithms_to_test: list[str],
                 population_sizes_to_test: list[int],
                 ps_budgets_per_run_to_test: list[int],
                 ps_budget: int,
                 custom_crowding_operators_to_test: list[bool]):
        self.pRef_sizes_to_test = pRef_sizes_to_test
        self.pRef_origin_methods = pRef_origin_methods
        self.island_amounts_to_test = island_amounts_to_test
        self.algorithms_to_test = algorithms_to_test
        self.population_sizes_to_test = population_sizes_to_test
        self.ps_budgets_per_run_to_test = ps_budgets_per_run_to_test
        self.ps_budget = ps_budget
        self.custom_crowding_operators_to_test = custom_crowding_operators_to_test




    def get_data(self):
        results = []
        for island_amount in self.island_amounts_to_test:
            gc_problem = GraphColouring.make_insular_instance(island_amount)
            bt_problem = EfficientBTProblem.from_Graph_Colouring(gc_problem)
            targets: set[PS] = gc_problem.get_targets()
            for pRef_origin_method in self.pRef_origin_methods:
                for pRef_size in self.pRef_sizes_to_test:
                    pRef = PRefManager.generate_pRef(problem = bt_problem,
                                                     which_algorithm= pRef_origin_method,
                                                     sample_size=pRef_size)
                    for miner_algorithm in self.algorithms_to_test:
                        for uses_custom_crowding_operator in self.custom_crowding_operators_to_test:
                            for population_size in self.population_sizes_to_test:
                                for ps_budget_per_run in self.ps_budgets_per_run_to_test:
                                    logger.info(f"{island_amount = }, "
                                          f"{pRef_origin_method = },"
                                          f"{pRef_size = }, "
                                          f"{miner_algorithm = }, "
                                          f"{population_size = }, "
                                          f"{ps_budget_per_run = },"
                                          f"{uses_custom_crowding_operator},"
                                          f"ps_budget = {self.ps_budget}")
                                    try:
                                        miner = SequentialCrowdingMiner(pRef = pRef,
                                                                        which_algorithm=miner_algorithm,
                                                                        population_size_per_run=population_size,
                                                                        budget_per_run=ps_budget_per_run,
                                                                        use_experimental_crowding_operator=uses_custom_crowding_operator)
                                        termination_criterion = TerminationCriteria.PSEvaluationLimit(self.ps_budget)
                                        with execution_time() as timer:
                                            miner.run(termination_criterion)
                                        mined_pss = miner.get_results()
                                        found = targets.intersection(mined_pss)
                                        datapoint = {"total_ps_budget": self.ps_budget,
                                                     "island_amount": island_amount,
                                                     "pRef_origin_method": pRef_origin_method,
                                                     "miner_algorith": miner_algorithm,
                                                     "population_size": population_size,
                                                     "ps_budget_per_run": ps_budget_per_run,
                                                     "mined_count": len(mined_pss),
                                                     "target_count": len(targets),
                                                     "found_count": len(found),
                                                     "uses_custom_crowding_operator": uses_custom_crowding_operator,
                                                     "runtime":timer.execution_time}
                                    except Exception as e:
                                        datapoint = {"ERROR": repr(e),
                                                     "total_ps_budget": self.ps_budget,
                                                     "island_amount": island_amount,
                                                     "pRef_origin_method": pRef_origin_method,
                                                     "miner_algorith": miner_algorithm,
                                                     "population_size": population_size,
                                                     "ps_budget_per_run": ps_budget_per_run,
                                                     "uses_custom_crowding_operator": uses_custom_crowding_operator
                                                     }
                                    results.append(datapoint)
        print(json.dumps(results, indent=4))

