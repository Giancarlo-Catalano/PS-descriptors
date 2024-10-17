import json

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairExplanationTester import PairExplanationTester


def consistency_test():
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  random_state=42)
    # problem = RoyalRoad(5)

    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=1000,
                                   ps_search_population=50,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=False)

    all_results = dict()

    for search_budget in [1000, 10000]:
        tester.ps_search_budget = search_budget
        for search_population_size in [50, 200]:
            print(f"Searching through {search_budget = }, {search_population_size}")
            tester.ps_search_population_size = search_population_size
            all_results[f"{search_budget}, {search_population_size}"] = tester.consistency_test_on_optima(runs=100,
                                                                                                          culling_method="overlap")

    file_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\messing_around\results_of_consistency_search.json"
    with open(file_path, "w") as file:
        json.dump(all_results, file)
    print(all_results)


def run_tester():
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  skills_to_use = {"woodworking", "fibre", "tech support", "electricity"},
                                                  random_state=42,
                                                  max_rota_length=3)
    # problem = RoyalRoad(5)


    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=2000,
                                   ps_search_population=100,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=True)


    #tester.get_random_explanation()
    #results = tester.consistency_test_on_optima(runs=100, culling_method=tester.preferred_culling_method)
    #results = tester.accuracy_test(amount_of_samples=100)
    #file_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\messing_around\results_of_accuracy_search_biggest.json"
    # with open(file_path, "w") as file:
    #     json.dump(results, file)
    # print(json.dumps(results))


    pretty_printer = BTProblemPrettyPrinter(descriptor_manager=None,
                                            pair_finder=tester,
                                            problem = problem)

    print("And the problem was ")
    def header(header_name: str):
        print(f"\n\n\n\n###{header_name}###")


    header("WORKERS")
    print(pretty_printer.repr_problem_workers())

    header("ROTAS")
    print(pretty_printer.repr_problem_rotas())

    ps = tester.get_explanation_to_improve_saturday()

    header("Partial Solution")
    print(pretty_printer.repr_partial_solution(ps))


run_tester()
