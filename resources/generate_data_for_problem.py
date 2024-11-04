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
from Explanation.PRefManager import PRefManager
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from resources.explanations.manage_explanations import ExplanationStorer
import utils

# this file stores all the information for a given problem, including
# problem definition, pRef, optima, explanations, descriptors and questions

skill_emoji_dict = {"electricity": "⚡",
                    "fibre": "📞",
                    "tech support": "💻",
                    "woodworking": "🔨",
                    "plumbing": "🔧"}


class ProblemInfoManager:
    problem: Optional[EfficientBTProblem]
    main_dir: str

    def __init__(self, problem, main_dir):
        self.problem = problem
        self.main_dir = main_dir


    @property
    def problem_path(self) -> str:
        return os.path.join(self.main_dir, "problem")

    @property
    def problem_json_path(self) -> str:
        return os.path.join(self.problem_path, "definition.json")

    @property
    def problem_tables_path(self) -> str:
        return os.path.join(self.problem_path, "tables.txt")

    @property
    def pRef_path(self) -> str:
        return os.path.join(self.main_dir, "pRef.npz")

    @property
    def explanations_path(self) -> str:
        return os.path.join(self.main_dir, "explanations")

    @property
    def descriptor_path(self) -> str:
        return os.path.join(self.explanations_path, "descriptor")

    def store_problem_json(self):
        problem_json = self.problem.to_json()
        with utils.open_and_make_directories(self.problem_json_path) as file:
            json.dump(problem_json, file, indent=4)

        print(f"Stored the problem json into the file {self.problem_json_path}")

    def load_problem(self):
        with open(self.problem_json_path, "r") as file:
            json_data = json.load(file)
        self.problem = EfficientBTProblem.from_json(json_data)
        return self.problem

    def make_pretty_printer(self) -> BTProblemPrettyPrinter:
        return BTProblemPrettyPrinter(descriptor_manager=None,
                                      problem=self.problem,
                                      skill_emoji_dict=skill_emoji_dict)

    def store_problem_visualisations(self):
        pretty_printer = self.make_pretty_printer()

        text_contents = "WORKERS for problem A\n"
        text_contents += pretty_printer.repr_problem_workers()
        text_contents += "\n" * 3

        text_contents += "ROTAS for problem A\n"
        text_contents += pretty_printer.repr_problem_rotas()

        with utils.open_and_make_directories(self.problem_tables_path) as text_file:
            text_file.write(text_contents)

        print(f"Wrote the problem tables onto file {self.problem_tables_path}")

    @property
    def conversion_json_path(self) -> str:
        return os.path.join(self.main_dir, "conversion.json")

    def generate_and_store_pRef(self):
        pRef = PRefManager.generate_pRef(problem=self.problem,
                                         which_algorithm="uniform GA",
                                         sample_size=10000)
        pRef.save(self.pRef_path)

    def load_pRef(self) -> PRef:
        return PRef.load(self.pRef_path)

    @property
    def optima_representation_path(self) -> str:
        return os.path.join(self.main_dir, "optima_representation.txt")

    def store_optima_visualisations(self):
        pRef = self.load_pRef()
        optima = pRef.get_best_solution()
        pretty_printer = self.make_pretty_printer()

        normal_representation = pretty_printer.repr_full_solution(optima)
        calendar = pretty_printer.get_calendar_counts_for_ps(PS.from_FS(optima))
        calendar_string = pretty_printer.repr_skill_calendar(calendar)
        penalties_strings = pretty_printer.get_penalties_string(calendar)
        fitness = self.problem.fitness_function(optima)
        textual_contents = "\n\n".join([normal_representation,
                                        calendar_string,
                                        penalties_strings,
                                        f"The optima is {fitness:.3f}"])

        with utils.open_and_make_directories(self.optima_representation_path) as optima_file:
            optima_file.write(textual_contents)

        print(f"Wrote the representation of the optima in {self.optima_representation_path}")

    def make_bootstrap_descriptor(self) -> DescriptorsManager:
        return DescriptorsManager.with_no_samples_yet(problem=self.problem,
                                                      control_samples_per_size_category=1000,
                                                      specialty_threshold=0.1,
                                                      verbose=False)


    def load_descriptor(self) -> DescriptorsManager:
        return DescriptorsManager.load(problem = self.problem, directory=self.explanations_path)

    def store_descriptor(self, descriptor: DescriptorsManager):
        descriptor.store(directory=self.descriptor_path)

    def explanation_text_folder_path(self, expl: PairwiseExplanation):
        return os.path.join(self.explanations_path, expl.label)

    def store_explanation(self,
                          expl: PairwiseExplanation,
                          explanation_manager: ExplanationStorer):
        explanation_folder = self.explanation_text_folder_path(expl)
        text_file_name = os.path.join(explanation_folder, "textual.txt")
        json_file_name = os.path.join(explanation_folder, "explanation.json")
        image_file_name = os.path.join(explanation_folder, "explanation.png")

        explanation_manager.store_single_explanation(expl, textual_path=text_file_name,
                                                     json_path=json_file_name,
                                                     image_path=image_file_name)

        print(f"Stored the details for an explanation at "
              f"{text_file_name = }, "
              f"{json_file_name = }, "
              f"{image_file_name = }")

    def get_explanation_manager(self, descriptor: Optional[DescriptorsManager] = None) -> ExplanationStorer:
        if descriptor is None:
            descriptor = self.make_bootstrap_descriptor()
        return ExplanationStorer(descriptor=descriptor,
                                 explanation_directory=self.explanations_path,
                                 pRef = self.load_pRef(),
                                 pretty_printer=self.make_pretty_printer(),
                                 problem = self.problem)

    def generate_and_store_explanations(self,
                                        explanation_manager: ExplanationStorer):
        indexes_to_compare_against = [1, 3, 5, 7, 9, 11, 13]

        pRef = explanation_manager.pRef
        best_solutions = pRef.get_top_n_solutions(max(indexes_to_compare_against)+1)
        optima = best_solutions[0]

        # generate the explanations
        explanations = [explanation_manager.generate_explanation(main_solution=optima,
                                                                 background_solution=best_solutions[index],
                                                                 label=f"Optima against solution[{index}]")
                        for index in tqdm(indexes_to_compare_against)]

        # store any changes to the descriptor
        explanation_manager.descriptor.store(self.descriptor_path)

        # store the actual explanations
        for expl in explanations:
            self.store_explanation(expl, explanation_manager)

    def store_everything(self):
        self.store_problem_json()
        self.store_problem_visualisations()

        self.generate_and_store_pRef()
        self.store_optima_visualisations()

        explanation_manager = self.get_explanation_manager()
        self.generate_and_store_explanations(explanation_manager)

def big_bang():
    seed = 42
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  skills_to_use={"woodworking", "fibre", "tech support",
                                                                 "electricity"},
                                                  random_state=seed,
                                                  max_rota_length=3,
                                                  calendar_length=8 * 7)

    main_dir = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\version_C"
    problem_manager = ProblemInfoManager(problem, main_dir)

    problem_manager.store_everything()


big_bang()

