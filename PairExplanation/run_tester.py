from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from PairExplanation.PairExplanationTester import PairExplanationTester


def run_tester():
    # problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
    #                                               quantity_workers_to_keep=30,
    #                                               random_state=42)
    problem = RoyalRoad(5)


    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=1000,
                                   ps_search_population=50,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=True)


    all_results = dict()

    for search_budget in [1000, 2000, 5000, 10000]:
        tester.ps_search_budget = search_budget
        for search_population_size in [50, 100, 200]:
            print(f"Searching through {search_budget = }, {search_population_size}")
            tester.ps_search_population_size = search_population_size
            all_results[(search_budget, search_population_size)] = tester.consistency_test_on_optima(runs=100, only_return_biggest=True)

    print(all_results)



run_tester()
