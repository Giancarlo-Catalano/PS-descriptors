from BenchmarkProblems.RoyalRoad import RoyalRoad
from PairExplanation.PairExplanationTester import PairExplanationTester


def run_tester():
    problem = RoyalRoad(4, 4)


    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=1000,
                                   ps_search_population=50,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=True)

    results = tester.consistency_test_on_optima(runs=100, only_return_biggest=True)
    print(results)



run_tester()
