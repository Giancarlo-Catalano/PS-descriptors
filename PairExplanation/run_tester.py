import json
import random

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
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
    seed = 42
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  skills_to_use={"woodworking", "fibre", "tech support", "electricity"},
                                                  random_state=seed,
                                                  max_rota_length=3)
    # problem = RoyalRoad(5)

    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=2000,
                                   ps_search_population=100,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=False)

    # tester.get_random_explanation()
    # results = tester.consistency_test_on_optima(runs=100, culling_method=tester.preferred_culling_method)
    # results = tester.accuracy_test(amount_of_samples=100)
    # file_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\messing_around\results_of_accuracy_search_biggest.json"
    # with open(file_path, "w") as file:
    #     json.dump(results, file)
    # print(json.dumps(results))

    descriptor = tester.get_temporary_descriptors_manager()
    pretty_printer = BTProblemPrettyPrinter(descriptor_manager=descriptor,
                                            problem=problem)


    print("And the problem was ")

    def header(header_name: str):
        print(f"\n\n\n\n###{header_name}###")

    header("WORKERS")
    print(pretty_printer.repr_problem_workers())

    header("ROTAS")
    print(pretty_printer.repr_problem_rotas())

    header("Optima FS")
    optima = tester.pRef.get_best_solution()
    print(pretty_printer.repr_full_solution(optima))

    header("Pairwise explanations")
    random.seed(seed)
    near_optimal_solutions = tester.pRef.get_top_n_solutions(30)
    amount_of_pairs_to_generate = 4
    main_solutions = random.sample(population=near_optimal_solutions, k=amount_of_pairs_to_generate)
    background_solutions = [tester.get_background_solutions(sol, 1)[0]
                            for sol in main_solutions]

    pairwise_explanations = [tester.get_pairwise_explanation(m, b, descriptor=descriptor)
                    for m, b in zip(main_solutions, background_solutions)]


    for explanation_item in pairwise_explanations:
        header("explanation item")
        explanation_item.print_using_pretty_printer(pretty_printer)


    header("Improving the weekdays")

    weekday_improvement_explanation = tester.get_explanation_to_improve_weekday(optima, "Tuesday", descriptor)

    header("Partial Solution")
    weekday_improvement_explanation.print_using_pretty_printer(pretty_printer)

    # header("Calendar for skill")
    # calendar = pretty_printer.get_calendar_counts_for_ps(ps)
    # print(pretty_printer.repr_skill_calendar(calendar))


run_tester()
