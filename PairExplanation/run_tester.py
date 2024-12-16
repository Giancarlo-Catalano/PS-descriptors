import json
import random

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima
from PairwiseExplanation.PRefManager import PRefManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from PairExplanation.ExplanationMiner import ExplanationMiner


def consistency_test():
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  random_state=42)
    # problem = RoyalRoad(5)

    tester = ExplanationMiner(optimisation_problem=problem,
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

    file_path = r"/resources/explanations/old_material/messing_around\results_of_consistency_search.json"
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

    pRef = PRefManager.generate_pRef(problem=problem,
                                         which_algorithm="uniform GA",
                                         sample_size=10000)
    # problem = RoyalRoad(5)

    tester = ExplanationMiner(optimisation_problem=problem,
                              ps_search_budget=2000,
                              ps_search_population=100,
                              pRef = pRef,
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

    def print_explanation(expl: PairwiseExplanation):
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
    # weekday_improvement_explanation = tester.get_explanation_to_improve_weekday(center_solution, "Tuesday", descriptors)
    #
    # header("Partial Solution")
    # for expl in [weekday_improvement_explanation]:
    #     print_explanation(expl)

    # header("Calendar for skill")
    # calendar = pretty_printer.get_calendar_counts_for_ps(ps)
    # print(pretty_printer.repr_skill_calendar(calendar))




def run_tester_on_RR():
    seed = 42
    problem = RoyalRoad(5)

    pRef = PRefManager.generate_pRef(problem=problem,
                                     which_algorithm="uniform GA",
                                     sample_size=10000)
    tester = ExplanationMiner(optimisation_problem=problem,
                              ps_search_budget=2000,
                              ps_search_population=100,
                              pRef=pRef,
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


    def print_explanation(expl: PairwiseExplanation):
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
    # weekday_improvement_explanation = tester.get_explanation_to_improve_weekday(center_solution, "Tuesday", descriptors)
    #
    # header("Partial Solution")
    # for expl in [weekday_improvement_explanation]:
    #     print_explanation(expl)

    # header("Calendar for skill")
    # calendar = pretty_printer.get_calendar_counts_for_ps(ps)
    # print(pretty_printer.repr_skill_calendar(calendar))


run_tester()
