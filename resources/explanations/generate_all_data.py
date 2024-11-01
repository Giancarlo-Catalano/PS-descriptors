import itertools
import json
import os
import random
from typing import Optional

import numpy as np
from tqdm import tqdm

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import STAR, PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.SearchSpace import SearchSpace
from Explanation.PRefManager import PRefManager
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.BakedPairwiseExplanation import BakedPairwiseExplanation
from PairExplanation.PairExplanationTester import PairExplanationTester
from PairExplanation.WeightedGraphVisualiser import WeightedGraphVisualiser
from utils import open_and_make_directories

skill_emoji_dict = {"electricity": "⚡",
                    "fibre": "📞",
                    "tech support": "💻",
                    "woodworking": "🔨",
                    "plumbing": "🔧"}

root = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\latest_material"
problem_A_path = os.path.join(root, "problem_A.json")
problem_B_path = os.path.join(root, "problem_B.json")

problem_representation_file_path = os.path.join(root, "problem_representations.txt")
pref_path = os.path.join(root, "pRef_A.npz")
pRef_B_path = os.path.join(root, "pRef_B.npz")

descriptors_folder = os.path.join(root, "descriptors")

optima_representation_file_path = os.path.join(root, "optima_representations.txt")

conversion_json_path = os.path.join(root, "conversion_A_to_B.json")

descriptor_path = os.path.join(root, "descriptor_A")
descriptor_B_folder = os.path.join(root, "descriptor_B")

explanation_folder = os.path.join(root, "explanations")
explanations_A_folder = os.path.join(explanation_folder, "explanations_A")
explanations_B_folder = os.path.join(explanation_folder, "explanations_B")


def store_problem_into_file(problem: EfficientBTProblem, path: str) -> None:
    problem_json = problem.to_json()
    with open_and_make_directories(path) as file:
        json.dump(problem_json, file, indent=4)

    print("Stored the problem into the file file")


def load_bt_problem_from_file(path: str) -> EfficientBTProblem:
    with open(path, "r") as file:
        json_data = json.load(file)
    return EfficientBTProblem.from_json(json_data)


def generate_problem_files():
    seed = 42
    problem_A = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                    quantity_workers_to_keep=30,
                                                    skills_to_use={"woodworking", "fibre", "tech support",
                                                                   "electricity"},
                                                    random_state=seed,
                                                    max_rota_length=3,
                                                    calendar_length=8 * 7)

    problem_B, conversion = EfficientBTProblem.make_secretly_identical_instance(problem_A)
    store_problem_into_file(problem_A, problem_A_path)
    store_problem_into_file(problem_B, problem_B_path)

    with open_and_make_directories(conversion_json_path) as file:
        json.dump(conversion, file, indent=4)

    print(f"Problem A was stored in {problem_A_path}")
    print(f"Problem B was stored in {problem_B_path}")
    print(f"The conversion between A and B is in {conversion_json_path}")


def header(header_name: str):
    print(f"\n\n\n\n###{header_name}###")


def generate_problem_tables():
    problem_A = load_bt_problem_from_file(problem_A_path)
    problem_B = load_bt_problem_from_file(problem_B_path)

    pretty_printer_A = BTProblemPrettyPrinter(descriptor_manager=None,
                                              problem=problem_A,
                                              skill_emoji_dict=skill_emoji_dict)

    pretty_printer_B = BTProblemPrettyPrinter(descriptor_manager=None,
                                              problem=problem_B,
                                              skill_emoji_dict=skill_emoji_dict)

    text_contents = ""

    text_contents += "WORKERS for problem A\n"
    text_contents += pretty_printer_A.repr_problem_workers()
    text_contents += "\n" * 3

    text_contents += "ROTAS for problem A\n"
    text_contents += pretty_printer_A.repr_problem_rotas()
    text_contents += "\n" * 6

    text_contents += "WORKERS for problem B\n"
    text_contents += pretty_printer_B.repr_problem_workers()
    text_contents += "\n" * 3

    text_contents += "ROTAS for problem A\n"
    text_contents += pretty_printer_B.repr_problem_rotas()

    with open_and_make_directories(problem_representation_file_path) as text_file:
        text_file.write(text_contents)

    print(f"Wrote the problem tables onto file {problem_representation_file_path}")


def generate_pRef_files():
    with open(conversion_json_path, "r") as conversion_file:
        conversion_dict = json.load(conversion_file)

    original_search_space_permutation = conversion_dict["worker_permutation_dict"]
    search_space_permutation = [original_search_space_permutation[str(index)]
                                for index in range(len(original_search_space_permutation))]

    problem_A = load_bt_problem_from_file(problem_A_path)
    pRef_A = PRefManager.generate_pRef(problem=problem_A,
                                       which_algorithm="uniform GA",
                                       sample_size=10000)

    old_cardinalities = np.array(problem_A.search_space.cardinalities)
    new_search_space = SearchSpace(old_cardinalities[search_space_permutation])

    new_full_solution_matrix = pRef_A.full_solution_matrix[:, search_space_permutation]

    pRef_B = PRef(fitness_array=pRef_A.fitness_array,
                  search_space=new_search_space,
                  full_solution_matrix=new_full_solution_matrix)

    pRef_A.save(pref_path)
    print(f"The pRef for problem A was stored in {pref_path}")

    pRef_B.save(pRef_B_path)
    print(f"The pRef for problem A was stored in {pRef_B_path}")


def generate_optima_representations():
    problem_A = load_bt_problem_from_file(problem_A_path)
    problem_B = load_bt_problem_from_file(problem_B_path)

    pRef_A = PRef.load(pref_path)
    pRef_B = PRef.load(pRef_B_path)

    def get_string_of_best(pRef: PRef, problem: EfficientBTProblem) -> str:
        optima = pRef.get_best_solution()
        pretty_printer = BTProblemPrettyPrinter(descriptor_manager=None,
                                                problem=problem,
                                                skill_emoji_dict=skill_emoji_dict)

        normal_representation = pretty_printer.repr_full_solution(optima)
        calendar = pretty_printer.get_calendar_counts_for_ps(PS.from_FS(optima))
        calendar_string = pretty_printer.repr_skill_calendar(calendar)
        penalties_strings = pretty_printer.get_penalties_string(calendar)
        fitness = problem.fitness_function(optima)
        return "\n\n".join([normal_representation, calendar_string, penalties_strings, f"The optima is {fitness:.3f}"])

    text_contents = ""

    text_contents += "Optima of Problem A\n"
    text_contents += get_string_of_best(pRef_A, problem_A)
    text_contents += "\n" * 3
    text_contents += "Optima of Problem B\n"
    text_contents += get_string_of_best(pRef_B, problem_B)

    with open_and_make_directories(optima_representation_file_path) as optima_file:
        optima_file.write(text_contents)

    print(f"Wrote the representation of the optima in {optima_representation_file_path}")


def generate_descriptors():
    problem_A = load_bt_problem_from_file(problem_A_path)
    problem_B = load_bt_problem_from_file(problem_B_path)

    def generate_descriptor_for_problem(problem: EfficientBTProblem, folder_name: str):
        descriptor = DescriptorsManager.with_no_samples_yet(problem=problem,
                                                            control_samples_per_size_category=1000,
                                                            specialty_threshold=0.1,
                                                            verbose=False)
        descriptor.store(folder_name)

    generate_descriptor_for_problem(problem_A, folder_name=descriptor_path)
    print(f"Stored the descriptor for problem A in {descriptor_path}")
    generate_descriptor_for_problem(problem_B, folder_name=descriptor_B_folder)
    print(f"Stored the descriptor for problem B in {descriptor_B_folder}")


def generate_explanations():
    def store_explanations_given_settings(problem_path,
                                          descriptor_path,
                                          pRef_path,
                                          indexes_to_compare_against,
                                          explanation_path: str):
        problem = load_bt_problem_from_file(problem_path)
        descriptor = DescriptorsManager.load(problem, directory=descriptor_path)
        pRef = PRef.load(pRef_path)
        tester = PairExplanationTester(optimisation_problem=problem,
                                       ps_search_budget=2000,
                                       ps_search_population=100,
                                       pRef=pRef,
                                       verbose=False)
        optima = pRef.get_best_solution()
        best_solutions = pRef.get_top_n_solutions(max(indexes_to_compare_against))
        background_solutions = [best_solutions[index] for index in indexes_to_compare_against]

        explanations = [tester.get_pairwise_explanation(optima,
                                                        b,
                                                        descriptor=descriptor)
                        for b in tqdm(background_solutions)]

        for expl, index_compared_against in zip(explanations, indexes_to_compare_against):
            expl.label = f"against solution[{index_compared_against}]"

        descriptor.store(descriptor_path)

        pretty_printer = BTProblemPrettyPrinter(problem, descriptor_manager=descriptor,
                                                skill_emoji_dict=skill_emoji_dict)


        linkage_learner = TraditionalPerturbationLinkage(problem)
        weighted_graph_visualiser = WeightedGraphVisualiser()

        store_explanations_as_json(explanations, explanation_path)
        store_explanations_as_text(explanations, explanation_path, pretty_printer)
        store_explanations_as_images(explanations, explanation_path, pretty_printer, linkage_learner, weighted_graph_visualiser)

        return explanations

    def store_explanations_as_json(explanations: list[BakedPairwiseExplanation], directory: str):
        pss_json = [expl.to_json() for expl in explanations]
        json_file_path = os.path.join(directory, "explanations.json")

        with open_and_make_directories(json_file_path) as json_file:
            json.dump(pss_json, json_file, indent=4)

        print(f"The explanations were stored in json form at in {json_file_path}")

    def store_explanations_as_text(explanations: list[BakedPairwiseExplanation],
                                   directory: str,
                                   pretty_printer: BTProblemPrettyPrinter):
        def get_textual_explanation(expl: BakedPairwiseExplanation):
            return "\n\n".join([f"label: {expl.label}",
                                expl.get_difference_in_rotas_table(pretty_printer),
                                expl.get_changes_in_range(pretty_printer),
                                expl.get_ps_table(pretty_printer),
                                expl.explanation_text])

        explanations_text = ("\n" * 5).join(map(get_textual_explanation, explanations))
        text_file_path = os.path.join(directory, "explanations.txt")

        with open_and_make_directories(text_file_path) as text_file:
            text_file.write(explanations_text)

        print(f"The explanations were stored in textual form in {text_file_path}")


    def store_explanations_as_images(explanations, directory, pretty_printer, linkage_learner, weighted_graph_visualiser):
        images_directory = os.path.join(directory, "visual")

        def store_linkage_image(expl: BakedPairwiseExplanation):
            file_name = os.path.join(images_directory, f"expl_{expl.label}.png")
            ps = expl.partial_solution
            print(f"This ps has {ps.fixed_count()} fixed variables")
            workers = pretty_printer.problem.workers
            names = [workers[index].name for index in ps.get_fixed_variable_positions()]
            linkage_table = linkage_learner.get_table_for_ps(ps)
            plot = weighted_graph_visualiser.make_plot(linkage_table, names)
            plot.savefig(file_name)

        for expl in explanations:
            store_linkage_image(expl)


    # todo call store explanations given settings



# generate_problem_files()
# generate_problem_tables()
# generate_pRef_files()
generate_optima_representations()
