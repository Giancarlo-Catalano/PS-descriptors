import json
import os
from typing import Optional

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from PairExplanation.ExplanationMiner import ExplanationMiner
from PairExplanation.WeightedGraphVisualiser import WeightedGraphVisualiser
from utils import open_and_make_directories


class ExplanationStorer:
    problem: EfficientBTProblem
    descriptor: DescriptorsManager
    pretty_printer: BTProblemPrettyPrinter
    pRef: PRef
    explanation_directory: str
    tester: ExplanationMiner
    linkage_learner: TraditionalPerturbationLinkage
    hypothesis_tester: Optional[WilcoxonTest]

    def __init__(self,
                 problem: EfficientBTProblem,
                 descriptor: DescriptorsManager,
                 pretty_printer: BTProblemPrettyPrinter,
                 hypothesis_tester: Optional[WilcoxonTest],
                 pRef: PRef,
                 explanation_directory: str):
        self.problem = problem
        self.descriptor = descriptor
        self.pretty_printer = pretty_printer
        self.pRef = pRef
        self.explanation_directory = explanation_directory
        self.tester = self.get_tester()
        self.linkage_learner = self.get_linkage_learner()
        self.weighted_graph_visualiser = self.get_graph_visualiser()
        self.hypothesis_tester = hypothesis_tester

    def get_tester(self):
        return ExplanationMiner(optimisation_problem=self.problem,
                                ps_search_budget=10000,
                                ps_search_population=100,
                                pRef=self.pRef,
                                verbose=False)

    def get_linkage_learner(self) -> TraditionalPerturbationLinkage:
        return TraditionalPerturbationLinkage(self.problem)

    def get_graph_visualiser(self) -> WeightedGraphVisualiser:
        return WeightedGraphVisualiser(only_show_percentage=0.2)

    def generate_explanation(self, main_solution: FullSolution,
                             background_solution: FullSolution,
                             label: str) -> PairwiseExplanation:
        expl = self.tester.get_pairwise_explanation(main_solution,
                                                    background_solution,
                                                    descriptor=self.descriptor)
        expl.label = label
        return expl

    def store_explanations(self, expls: list[PairwiseExplanation]):
        json_destination = os.path.join(self.explanation_directory, "explanations.json")
        text_destination = os.path.join(self.explanation_directory, "explanations.txt")
        image_destination_folder = os.path.join(self.explanation_directory, "linkage_images")
        self.store_explanations_as_json(expls, json_destination)
        self.store_explanations_as_text(expls, text_destination)
        self.store_explanations_as_images(expls, image_destination_folder)

    def get_textual_explanation(self, expl: PairwiseExplanation):
        return "\n\n".join([f"label: {expl.label}",
                            expl.get_difference_in_rotas_table(self.pretty_printer),
                            expl.get_changes_in_range(self.pretty_printer),
                            expl.get_ps_table(self.pretty_printer),
                            expl.explanation_text,
                            expl.get_hypothesis_test_results(self.hypothesis_tester)])

    def store_linkage_image(self, expl: PairwiseExplanation, file_name: str):
        ps = expl.partial_solution
        workers = self.pretty_printer.problem.workers
        names = [workers[index].name for index in ps.get_fixed_variable_positions()]
        self.linkage_learner.set_solution(expl.main_solution)
        linkage_table = self.linkage_learner.get_table_for_ps(ps)
        plot = self.weighted_graph_visualiser.make_plot(linkage_table, names)
        plot.savefig(file_name)

    def store_single_explanation(self, expl: PairwiseExplanation,
                                 textual_path: str,
                                 json_path: str,
                                 image_path: str):

        with open_and_make_directories(textual_path) as text_file:
            textual = self.get_textual_explanation(expl)
            text_file.write(textual)

        with open_and_make_directories(json_path) as json_file:
            json_data = expl.to_json()
            json.dump(json_data, json_file, indent=4)

        self.store_linkage_image(expl, image_path)



    def store_explanations_as_json(self, expls: list[PairwiseExplanation], json_destination: str):
        pss_json = [expl.to_json() for expl in expls]

        with open_and_make_directories(json_destination) as json_file:
            json.dump(pss_json, json_file, indent=4)

        print(f"The explanations were stored in json form at in {json_destination}")

    def store_explanations_as_text(self, expls: list[PairwiseExplanation], text_destination: str):
        def get_textual_explanation(expl: PairwiseExplanation):
            return "\n\n".join([f"label: {expl.label}",
                                expl.get_difference_in_rotas_table(self.pretty_printer),
                                expl.get_changes_in_range(self.pretty_printer),
                                expl.get_ps_table(self.pretty_printer),
                                expl.explanation_text])

        explanations_text = ("\n" * 5).join(map(get_textual_explanation, expls))

        with open_and_make_directories(text_destination) as text_file:
            text_file.write(explanations_text)

        print(f"The explanations were stored in textual form in {text_destination}")

    def store_explanations_as_images(self, expls: list[PairwiseExplanation], image_destination_folder: str):
        def store_linkage_image(expl: PairwiseExplanation):
            file_name = os.path.join(image_destination_folder, f"expl_{expl.label}.png")
            ps = expl.partial_solution
            print(f"This ps has {ps.fixed_count()} fixed variables")
            workers = self.pretty_printer.problem.workers
            names = [workers[index].name for index in ps.get_fixed_variable_positions()]
            linkage_table = self.linkage_learner.get_table_for_ps(ps)
            plot = self.weighted_graph_visualiser.make_plot(linkage_table, names)
            plot.savefig(file_name)

        for expl in expls:
            store_linkage_image(expl)

    def convert_explanation(self, original: PairwiseExplanation, conversion_data: dict) -> PairwiseExplanation:
        original_search_space_permutation = conversion_data["worker_permutation_dict"]
        search_space_permutation = [original_search_space_permutation[index]
                                    for index in range(len(original_search_space_permutation))]

        converted_main_solution = FullSolution(original.main_solution.values[search_space_permutation])
        converted_back_solution = FullSolution(original.main_solution.values[search_space_permutation])
        converted_ps = PS(original.partial_solution.values[search_space_permutation])

        new_explanation = PairwiseExplanation(converted_main_solution,
                                              converted_back_solution,
                                              partial_solution=converted_ps,
                                              descriptor_tuples=original.descriptor_tuples,
                                              explanation_text=original.explanation_text)
        new_explanation.label = original.label
        return new_explanation






