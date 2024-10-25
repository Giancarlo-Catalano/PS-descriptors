import itertools
import json
import random
from typing import Optional

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima
from Explanation.PRefManager import PRefManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.BakedPairwiseExplanation import BakedPairwiseExplanation
from PairExplanation.PairExplanationTester import PairExplanationTester
from utils import announce

json_file = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\PairExplanation\everything.json"
pRef_file = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\PairExplanation\pRef.npz"

seed = 42
problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                              quantity_workers_to_keep=30,
                                              skills_to_use={"woodworking", "fibre", "tech support", "electricity"},
                                              random_state=seed,
                                              max_rota_length=3,
                                              calendar_length=8 * 7)


def header(header_name: str):
    print(f"\n\n\n\n###{header_name}###")


def explanation_is_correct(expl, expl_generator, hypothesis_tester, near_optima_hypothesis_tester):
    assessment = expl_generator.evaluate_explanation(expl, hypothesis_tester, near_optima_hypothesis_tester)
    return assessment["is_accurate"]


def print_explanation(expl: BakedPairwiseExplanation,
                      pretty_printer, hypothesis_tester: Optional,
                      near_optima_hypothesis_tester: Optional):

    print(f"label = {expl.label}")
    expl.print_using_pretty_printer(pretty_printer, show_solutions=False,
                                    hypothesis_tester=hypothesis_tester,
                                    near_optima_hypothesis_tester=near_optima_hypothesis_tester)
    # is_correct = explanation_is_correct(expl)
    # print(f"{is_correct = }")


def generate_pRef():
    pRef = PRefManager.generate_pRef(problem=problem,
                                     which_algorithm="uniform GA",
                                     sample_size=10000)

    pRef.save(pRef_file)
    print(f"The pRef was stored in {pRef_file}")


def generate_explanations(pRef: PRef):
    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=2000,
                                   ps_search_population=100,
                                   pRef=pRef,
                                   verbose=False)

    descriptor = tester.get_temporary_descriptors_manager(control_samples_per_size_category=1)

    pretty_printer = BTProblemPrettyPrinter(descriptor_manager=descriptor,
                                            problem=problem)

    hypothesis_tester = WilcoxonTest(sample_size=1000,
                                     search_space=problem.search_space,
                                     fitness_evaluator=tester.fs_evaluator)
    near_optima_hypothesis_tester = WilcoxonNearOptima(pRef=tester.pRef,
                                                       evaluator=tester.fs_evaluator,
                                                       samples_required=100)

    header("WORKERS")
    print(pretty_printer.repr_problem_workers())

    header("ROTAS")
    print(pretty_printer.repr_problem_rotas())

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

    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        expl.label = f"main = 5, back = {background_index}"

    from_other_pairwise_explanations = [tester.get_pairwise_explanation(b,
                                                                        center_solution,
                                                                        descriptor=descriptor)
                                        for b in background_solutions]

    for expl, background_index in zip(from_other_pairwise_explanations, background_indexes):
        expl.label = f"main = {background_index}, back = 5"

    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of MAIN, compared to {background_index}")
        print_explanation(expl, pretty_printer, hypothesis_tester, near_optima_hypothesis_tester)

    for expl, background_index in zip(from_other_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of {background_index}, compared to MAIN")
        print_explanation(expl, pretty_printer, hypothesis_tester, near_optima_hypothesis_tester)

    pss_json = [expl.to_json() for expl in
                itertools.chain(from_main_pairwise_explanations, from_other_pairwise_explanations)]

    with open(json_file, "w") as pss_output_file:
        json.dump(pss_json, pss_output_file, indent=4)

    print(f"The explanations were stored in {pss_output_file}")


def load_from_json():
    pRef = PRef.load(pRef_file)

    tester = PairExplanationTester(optimisation_problem=problem,
                                   ps_search_budget=2000,
                                   ps_search_population=100,
                                   pRef=pRef,
                                   verbose=False)

    descriptor = tester.get_temporary_descriptors_manager(control_samples_per_size_category=1)
    pretty_printer = BTProblemPrettyPrinter(problem, descriptor_manager=descriptor)
    hypothesis_tester = WilcoxonTest(sample_size=1000,
                                     search_space=problem.search_space,
                                     fitness_evaluator=tester.fs_evaluator)
    near_optima_hypothesis_tester = WilcoxonNearOptima(pRef=tester.pRef,
                                                       evaluator=tester.fs_evaluator,
                                                       samples_required=100)

    with open(json_file, "r") as json_fid:
        expls_jsons = json.load(json_fid)

    expls = [BakedPairwiseExplanation.from_json(expl_json) for expl_json in expls_jsons]

    for expl in expls:
        print_explanation(expl,
                          pretty_printer,
                          hypothesis_tester,
                          near_optima_hypothesis_tester)



# generate_pRef()
# pRef = PRef.load(pRef_file)
# generate_explanations(pRef)


load_from_json()
