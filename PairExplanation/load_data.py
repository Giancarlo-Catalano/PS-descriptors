import itertools
import json
import os
import random
from typing import Optional

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Core.PS import STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from PairwiseExplanation.PRefManager import PRefManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from PairExplanation.ExplanationMiner import ExplanationMiner
from PairExplanation.WeightedGraphVisualiser import WeightedGraphVisualiser
from utils import announce

json_file = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\PairExplanation\everything.json"
pRef_file = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\PairExplanation\pRef.npz"

skill_emoji_dict = {"electricity": "⚡",
                    "fibre": "📞",
                    "tech support": "💻",
                    "woodworking": "🔨",
                    "plumbing": "🔧"}

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


def print_explanation(expl: PairwiseExplanation,
                      pretty_printer, hypothesis_tester: Optional,
                      near_optima_hypothesis_tester: Optional):
    print(f"label = {expl.label}")
    # expl.print_using_pretty_printer(pretty_printer, show_solutions=False,
    #                                 hypothesis_tester=hypothesis_tester,
    #                                 near_optima_hypothesis_tester=near_optima_hypothesis_tester)

    print("\n")

    print(expl.get_difference_in_rotas_table(pretty_printer))
    # print(expl.get_changes_in_calendar(pretty_printer))
    print(expl.get_changes_in_range(pretty_printer))
    # is_correct = explanation_is_correct(expl)
    print(expl.get_ps_table(pretty_printer))
    # print(f"{is_correct = }")

    print(expl.explanation_text)


def generate_pRef():
    pRef = PRefManager.generate_pRef(problem=problem,
                                     which_algorithm="uniform GA",
                                     sample_size=10000)

    pRef.save(pRef_file)
    print(f"The pRef was stored in {pRef_file}")


def generate_explanations(pRef: PRef):
    tester = ExplanationMiner(optimisation_problem=problem,
                              ps_search_budget=2000,
                              ps_search_population=100,
                              pRef=pRef,
                              verbose=False)

    descriptor = tester.get_temporary_descriptors_manager(control_samples_per_size_category=1000)

    pretty_printer = BTProblemPrettyPrinter(descriptor_manager=descriptor,
                                            problem=problem,
                                            skill_emoji_dict={"electricity": "⚡",
                                                              "fibre": "📞",
                                                              "tech support": "💻",
                                                              "woodworking": "🔨",
                                                              "plumbing": "🔧"})

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
    best_n_solutions = tester.pRef.get_top_n_solutions(16)
    center_solution = best_n_solutions[0]
    print(problem.repr_full_solution(center_solution))
    print(f"It has fitness {center_solution.fitness}")

    header("Pairwise explanations")
    random.seed(seed)

    background_indexes = list(range(1, 15))
    background_solutions = [best_n_solutions[index] for index in
                            background_indexes]  # before 5 is better, after 5 is worse

    from_main_pairwise_explanations = [tester.get_pairwise_explanation(center_solution,
                                                                       b,
                                                                       descriptor=descriptor)
                                       for b in background_solutions]

    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        expl.label = f"optima vs solution[{background_index}]"

    for expl, background_index in zip(from_main_pairwise_explanations, background_indexes):
        header(f"explanation item, it was a subset of MAIN, compared to {background_index}")
        print_explanation(expl, pretty_printer, hypothesis_tester, near_optima_hypothesis_tester)

    pss_json = [expl.to_json() for expl in from_main_pairwise_explanations]

    with open(json_file, "w") as pss_output_file:
        json.dump(pss_json, pss_output_file, indent=4)

    print(f"The explanations were stored in {pss_output_file}")


def store_textual_explanation(expl: PairwiseExplanation,
                              destination: str,
                              pretty_printer: BTProblemPrettyPrinter):
    text_to_be_stored = ""
    text_to_be_stored += expl.get_difference_in_rotas_table(pretty_printer)
    text_to_be_stored += "\n\n"
    text_to_be_stored += expl.get_changes_in_range(pretty_printer)
    text_to_be_stored += "\n\n"
    text_to_be_stored += expl.get_ps_table(pretty_printer)
    text_to_be_stored += "\n\n"
    text_to_be_stored += expl.explanation_text

    with open(destination, "w", encoding="utf-8") as file:
        file.write(text_to_be_stored)


def store_linkage_image(expl: PairwiseExplanation,
                        destination: str,
                        pretty_printer: BTProblemPrettyPrinter,
                        linkage_learner: TraditionalPerturbationLinkage,
                        weighted_graph_visualiser: WeightedGraphVisualiser):
    ps = expl.partial_solution
    print(f"This ps has {ps.fixed_count()} fixed variables")
    workers = pretty_printer.problem.workers
    names = [workers[index].name for index in ps.get_fixed_variable_positions()]
    linkage_table = linkage_learner.get_table_for_ps(ps)
    plot = weighted_graph_visualiser.make_plot(linkage_table, names)
    plot.savefig(destination)


def store_explanation(expl: PairwiseExplanation,
                      pretty_printer: BTProblemPrettyPrinter,
                      linkage_learner: TraditionalPerturbationLinkage,
                      weighted_graph_visualiser: WeightedGraphVisualiser):
    root = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\explanation_sheets"
    text_file_name = f"text_{expl.label}.txt"
    image_file_name = f"image_{expl.label}.png"

    text_destination = os.path.join(root, text_file_name)
    image_destination = os.path.join(root, image_file_name)

    store_textual_explanation(expl, destination=text_destination,
                              pretty_printer = pretty_printer)

    store_linkage_image(expl, destination=image_destination,
                        linkage_learner=linkage_learner,
                        weighted_graph_visualiser=weighted_graph_visualiser,
                        pretty_printer=pretty_printer)


def load_from_json():
    pRef = PRef.load(pRef_file)

    tester = ExplanationMiner(optimisation_problem=problem,
                              ps_search_budget=2000,
                              ps_search_population=100,
                              pRef=pRef,
                              verbose=False)

    descriptor = tester.get_temporary_descriptors_manager(control_samples_per_size_category=1000)
    pretty_printer = BTProblemPrettyPrinter(problem,
                                            descriptor_manager=descriptor,
                                            skill_emoji_dict=skill_emoji_dict)

    hypothesis_tester = WilcoxonTest(sample_size=1000,
                                     search_space=problem.search_space,
                                     fitness_evaluator=tester.fs_evaluator)
    near_optima_hypothesis_tester = WilcoxonNearOptima(pRef=tester.pRef,
                                                       evaluator=tester.fs_evaluator,
                                                       samples_required=100)

    linkage_learner = TraditionalPerturbationLinkage(problem)
    graph_visualiser = WeightedGraphVisualiser()


    with open(json_file, "r") as json_fid:
        expls_jsons = json.load(json_fid)

    expls = [PairwiseExplanation.from_json(expl_json) for expl_json in expls_jsons]

    for expl in expls:
        print_explanation(expl,
                          pretty_printer,
                          hypothesis_tester,
                          near_optima_hypothesis_tester)
        linkage_learner.set_solution(expl.main_solution)
        store_explanation(expl,
                          pretty_printer,
                          linkage_learner,
                          graph_visualiser)


# generate_pRef()
# pRef = PRef.load(pRef_file)
# generate_explanations(pRef)


# load_from_json()


def store_problem_into_file(problem: EfficientBTProblem, path: str) -> None:
    problem_json = problem.to_json()
    with open(path, "w") as file:
        json.dump(problem_json, file, indent=4)

    print("Stored the problem into the file file")


def load_bt_problem_from_file(path: str) -> EfficientBTProblem:
    with open(path, "r") as file:
        json_data = json.load(file)
    return EfficientBTProblem.from_json(json_data)




def store_normal_problem():
    destination = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\problems\problem_A.json"
    store_problem_into_file(problem, destination)

def make_two_secretly_identical_problems():
    root = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\problems"
    problem_A_path = os.path.join(root, "problem_A.json")
    problem_B_path = os.path.join(root, "problem_B.json")

    conversion_json_path = os.path.join(root, "conversion_A_to_B.json")

    problem_A = load_bt_problem_from_file(problem_A_path)
    problem_B, conversion = EfficientBTProblem.make_secretly_identical_instance(problem_A)
    store_problem_into_file(problem_B, problem_B_path)

    with open(conversion_json_path, "w") as file:
        json.dump(conversion, file, indent=4)


make_two_secretly_identical_problems()




