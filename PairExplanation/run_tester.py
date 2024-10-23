import json
import random

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.BakedPairwiseExplanation import BakedPairwiseExplanation
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
                                                  max_rota_length=3,
                                                  calendar_length=8 * 7)
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

    hypothesis_tester = WilcoxonTest(sample_size=1000,
                                     search_space=problem.search_space,
                                     fitness_evaluator=tester.fs_evaluator)
    near_optima_hypothesis_tester = WilcoxonNearOptima(pRef=tester.pRef,
                                                       evaluator=tester.fs_evaluator,
                                                       samples_required=100)

    print("And the problem was ")

    def header(header_name: str):
        print(f"\n\n\n\n###{header_name}###")

    def explanation_is_correct(expl):
        assessment = tester.evaluate_explanation(expl, hypothesis_tester, near_optima_hypothesis_tester)
        return assessment["is_accurate"]

    def print_explanation(expl: BakedPairwiseExplanation):
        expl.print_using_pretty_printer(pretty_printer, show_solutions=False,
                                        hypothesis_tester=hypothesis_tester,
                                        near_optima_hypothesis_tester=near_optima_hypothesis_tester)
        is_correct = explanation_is_correct(expl)
        print(f"{is_correct = }")

    header("WORKERS")
    print(pretty_printer.repr_problem_workers())

    header("ROTAS")
    print(pretty_printer.repr_problem_rotas())

    header("Main FS")
    best_n_solutions = tester.pRef.get_top_n_solutions(10)
    center_solution = best_n_solutions[5]
    print(pretty_printer.repr_full_solution(center_solution))
    print(pretty_printer.repr_extra_information_for_full_solution(center_solution))

    center_fitness = problem.fitness_function(center_solution)

    header("Pairwise explanations")
    random.seed(seed)

    background_indexes = [0, 3, 7, 9]
    background_solutions = [best_n_solutions[index] for index in
                            background_indexes]  # before 5 is better, after 5 is worse

    from_main_pairwise_explanations = [tester.get_pairwise_explanation(center_solution,
                                                                       b,
                                                                       descriptor=descriptor)
                                       for b in background_solutions]

    from_other_pairwise_explanations = [tester.get_pairwise_explanation(b,
                                                                        center_solution,
                                                                        descriptor=descriptor)
                                        for b in background_solutions]

    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of MAIN, compared to {background_index}")
        print_explanation(expl)

    for expl, background_index in zip(from_other_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of {background_index}, compared to MAIN")
        print_explanation(expl)

    # header("Improving the weekdays")
    #
    # weekday_improvement_explanation = tester.get_explanation_to_improve_weekday(center_solution, "Tuesday", descriptor)
    #
    # header("Partial Solution")
    # for expl in [weekday_improvement_explanation]:
    #     print_explanation(expl)

    # header("Calendar for skill")
    # calendar = pretty_printer.get_calendar_counts_for_ps(ps)
    # print(pretty_printer.repr_skill_calendar(calendar))

    """
    "max = 18, 
min = 12, 
p = 0.11"			"max = 25, 
min = 21, 
p = 0.03"			"max = 26, 
min = 25, 
p = 0.00"			"max = 23, min = 22, 
p = 0.00"			"max = 19, min = 19, 
p = 0.00"			"max = 14, min = 7, 
p = 0.25"			"max = 6, min = 5, 
p = 0.03"		
"max = 16, min = 10, 
p = 0.14"			"max = 19, min = 16, 
p = 0.02"			"max = 20, min = 19, 
p = 0.00"			"max = 18, min = 16, 
p = 0.01"			"max = 15, min = 13, 
p = 0.02"			"max = 10, min = 5, 
p = 0.25"			"max = 5, min = 4,
p = 0.04"		
"max = 5, min = 3, 
p = 0.16"			"max = 5, min = 4, 
p = 0.04"			"max = 6, min = 6, 
p = 0.00"			"max = 5, min = 4, 
p = 0.04"			"max = 3, min = 3, 
p = 0.00"			"max = 3, min = 1, 
p = 0.44"			"max = 2, min = 2, 
p = 0.00"		
"max = 6, min = 6, 
p = 0.00"			"max = 7, min = 7, 
p = 0.00"			"max = 8, min = 8, 
p = 0.00"			"max = 8, min = 8, 
p = 0.00"			"max = 4, min = 4, 
p = 0.00"			"max = 0, min = 0, 
p = 1.00"			"max = 3, 
min = 3, 
p = 0.00"		
    
    """


def run_tester_on_RR():
    seed = 42
    problem = RoyalRoad(5)

    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=2000,
                                   ps_search_population=100,
                                   pRef_size=10000,
                                   pRef_creation_method="uniform GA",
                                   verbose=False)

    descriptor = tester.get_temporary_descriptors_manager(control_samples_per_size_category=1)

    hypothesis_tester = WilcoxonTest(sample_size=1000,
                                     search_space=problem.search_space,
                                     fitness_evaluator=tester.fs_evaluator)
    near_optima_hypothesis_tester = WilcoxonNearOptima(pRef=tester.pRef,
                                                       evaluator=tester.fs_evaluator,
                                                       samples_required=100)

    print("And the problem was ")

    def header(header_name: str):
        print(f"\n\n\n\n###{header_name}###")

    def explanation_is_correct(expl):
        assessment = tester.evaluate_explanation(expl, hypothesis_tester, near_optima_hypothesis_tester)
        return assessment["is_accurate"]


    def print_explanation(expl: BakedPairwiseExplanation):
        expl.print_normally(problem,
                            show_solutions=True,
                            hypothesis_tester=hypothesis_tester,
                            near_optima_hypothesis_tester=near_optima_hypothesis_tester)

        is_correct = explanation_is_correct(expl)
        print(f"{is_correct = }")

    header("PROBLEM")
    print(problem)

    header("Main FS")
    best_n_solutions = tester.pRef.get_top_n_solutions(10)
    center_solution = best_n_solutions[5]
    print(problem.repr_full_solution(center_solution))
    print(f"It has fitness {center_solution.fitness}")

    header("Pairwise explanations")
    random.seed(seed)

    background_indexes = [0, 3, 7, 9]
    background_solutions = [best_n_solutions[index] for index in
                            background_indexes]  # before 5 is better, after 5 is worse

    from_main_pairwise_explanations = [tester.get_pairwise_explanation(center_solution,
                                                                       b,
                                                                       descriptor=descriptor)
                                       for b in background_solutions]

    from_other_pairwise_explanations = [tester.get_pairwise_explanation(b,
                                                                        center_solution,
                                                                        descriptor=descriptor)
                                        for b in background_solutions]



    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of MAIN, compared to {background_index}")
        print_explanation(expl)

    for expl, background_index in zip(from_other_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of {background_index}, compared to MAIN")
        print_explanation(expl)

    # header("Improving the weekdays")
    #
    # weekday_improvement_explanation = tester.get_explanation_to_improve_weekday(center_solution, "Tuesday", descriptor)
    #
    # header("Partial Solution")
    # for expl in [weekday_improvement_explanation]:
    #     print_explanation(expl)

    # header("Calendar for skill")
    # calendar = pretty_printer.get_calendar_counts_for_ps(ps)
    # print(pretty_printer.repr_skill_calendar(calendar))


run_tester()