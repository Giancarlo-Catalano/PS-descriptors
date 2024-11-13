import json
import json
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest
from Core.SearchSpace import SearchSpace
from Explanation.PRefManager import PRefManager
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter
from PairExplanation.PairwiseExplanation import PairwiseExplanation
from resources.explanations.manage_explanations import ExplanationStorer

# this file stores all the information for a given problem, including
# problem definition, pRef, optima, explanations, descriptors and questions

skill_emoji_dict = {"electricity": "⚡",
                    "fibre": "📞",
                    "tech support": "💻",
                    "woodworking": "🔨",
                    "plumbing": "🔧"}

current_expl_directory = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\Version_E"
problem_A_path = os.path.join(current_expl_directory, "problem_A")
problem_B_path = os.path.join(current_expl_directory, "problem_B")
example_problem_path = os.path.join(current_expl_directory, "example_problem")


class QuestionnaireDataForProblemGenerator:
    problem: Optional[EfficientBTProblem]
    main_dir: str

    def __init__(self, problem = None, main_dir = "invalid"):
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
        return DescriptorsManager.load(problem=self.problem, directory=self.explanations_path)

    def store_descriptor(self, descriptor: DescriptorsManager):
        descriptor.store(directory=self.descriptor_path)

    def get_specific_explanation_path(self, expl: PairwiseExplanation):
        return os.path.join(self.explanations_path, expl.label)

    def get_filenames_for_explanation(self, expl) -> (str, str, str):
        explanation_folder = self.get_specific_explanation_path(expl)
        text_file_name = os.path.join(explanation_folder, "textual.txt")
        json_file_name = os.path.join(explanation_folder, "explanation.json")
        image_file_name = os.path.join(explanation_folder, "explanation.png")
        return json_file_name, text_file_name, image_file_name

    def store_explanation(self,
                          expl: PairwiseExplanation,
                          explanation_manager: ExplanationStorer):
        json_file_name, text_file_name, image_file_name = self.get_filenames_for_explanation(expl)

        explanation_manager.store_single_explanation(expl, textual_path=text_file_name,
                                                     json_path=json_file_name,
                                                     image_path=image_file_name)

        print(f"Stored the details for an explanation at "
              f"{text_file_name = }, "
              f"{json_file_name = }, "
              f"{image_file_name = }")

    def load_explanation_from_folder(self, expl_folder_path: str, expl_label: str):
        # we just use the json
        json_file_name = os.path.join(expl_folder_path, "explanation.json")
        with open(json_file_name, "r") as file:
            json_data = json.load(file)
            expl = PairwiseExplanation.from_json(json_data)
            expl.label = expl_label
            return expl

    def load_explanations(self) -> list[PairwiseExplanation]:
        # the only stuff that we need is in the json
        # we get the label for the explanations from the folder names
        explanations_path = self.explanations_path
        folders = [(name, os.path.join(explanations_path, name))
                   for name in os.listdir(explanations_path)]
        descriptor_path = self.descriptor_path
        folders = [(folder_name, path) for folder_name, path in folders
                   if os.path.isdir(path)
                   if path != descriptor_path]

        explanations = [self.load_explanation_from_folder(path, label)
                        for label, path in folders]

        print(f"The explanations were loaded from the {explanations_path} folder, "
              f"with the following labels: " + (", ".join(utils.unzip(folders)[0])))
        return explanations

    def make_hypothesis_tester(self):
        fs_evaluator = FSEvaluator(self.problem.fitness_function)
        hypothesis_tester = WilcoxonTest(sample_size=1000,
                                         search_space=self.problem.search_space,
                                         fitness_evaluator=fs_evaluator)
        return hypothesis_tester

    def get_explanation_manager(self, descriptor: Optional[DescriptorsManager] = None) -> ExplanationStorer:
        if descriptor is None:
            descriptor = self.make_bootstrap_descriptor()

        return ExplanationStorer(descriptor=descriptor,
                                 explanation_directory=self.explanations_path,
                                 pRef=self.load_pRef(),
                                 pretty_printer=self.make_pretty_printer(),
                                 hypothesis_tester=self.make_hypothesis_tester(),
                                 problem=self.problem)

    def reload_and_store_explanations(self, explanations_manager: ExplanationStorer):
        print("Reloading and storing the explanations")
        explanations = self.load_explanations()
        for expl in explanations:
            self.store_explanation(expl, explanations_manager)

    def generate_and_store_explanations(self,
                                        explanation_manager: ExplanationStorer):
        indexes_to_compare_against = [1, 2, 3, 5, 10, 20, 50, 100, 1000]

        pRef = explanation_manager.pRef
        best_solutions = pRef.get_top_n_solutions(max(indexes_to_compare_against) + 1)
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

    def store_everything(self, generate_explanations_ex_novo: bool):
        self.store_problem_json()
        self.store_problem_visualisations()

        self.generate_and_store_pRef()  # note that this is overrriden in the permuted version
        self.store_optima_visualisations()

        explanation_manager = self.get_explanation_manager()
        if generate_explanations_ex_novo:
            self.generate_and_store_explanations(explanation_manager)
        else:
            self.reload_and_store_explanations(explanation_manager)


    def store_answers(self,
                      question_1_options: list,
                      question_2_options: list):
        pRef = self.load_pRef()
        optima = pRef.get_best_solution()
        def solution_if_worker_on_option(solution: FullSolution,
                                         worker_name: str,
                                         new_rota_choice: str):
            worker_index = [index for index, worker in enumerate(self.problem.workers)
                            if worker.name == worker_name][0]
            value = utils.alphabet.index(new_rota_choice)
            return solution.with_different_value(variable_index=worker_index, new_value=value)

        def get_fitness_for_modification(option: list[(str, str)]):
            current_solution = optima
            for worker, new_rota in option:
                current_solution = solution_if_worker_on_option(current_solution, worker, new_rota)
            return self.problem.fitness_function(current_solution)


        def arrange_modifications_by_entry(modifications):
            mods_and_fits = [(modif, get_fitness_for_modification(modif)) for modif in modifications]
            mods_and_fits.sort(key=utils.second, reverse=True)
            for index, (modif, fitness) in enumerate(mods_and_fits):
                print(f"#{index+1} -> {modif} -> penalty = {-fitness}")


        print("Results for question 1")
        arrange_modifications_by_entry(question_1_options)

        print("Results for question 2")
        arrange_modifications_by_entry(question_2_options)


def generate_for_first_problem(generate_explanations_ex_novo: bool):
    seed = 42
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  skills_to_use={"woodworking", "fibre", "tech support",
                                                                 "electricity"},
                                                  random_state=seed,
                                                  max_rota_length=3,
                                                  calendar_length=8 * 7)

    problem_manager = QuestionnaireDataForProblemGenerator(problem, problem_A_path)

    problem_manager.store_everything(generate_explanations_ex_novo)


def generate_for_example_problem(generate_explanations_ex_novo):
    seed = 6
    problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                  quantity_workers_to_keep=30,
                                                  skills_to_use={"woodworking", "fibre", "tech support",
                                                                 "electricity"},
                                                  random_state=seed,
                                                  max_rota_length=3,
                                                  calendar_length=8 * 7)

    problem_manager = QuestionnaireDataForProblemGenerator(problem, example_problem_path)

    problem_manager.store_everything(generate_explanations_ex_novo)


class QuestionnaireDataForPermutedProblemGenerator(QuestionnaireDataForProblemGenerator):
    original_problem_manager: QuestionnaireDataForProblemGenerator
    conversion_data: Optional[dict]

    def __init__(self,
                 original_problem_folder: str,
                 own_problem_folder: str):
        super().__init__(problem=None, main_dir=own_problem_folder)
        self.original_problem_manager = QuestionnaireDataForProblemGenerator(problem=None,
                                                                             main_dir=original_problem_folder)
        self.conversion_data = None

    def load_original_problem(self):
        self.original_problem_manager.load_problem()
        print(f"Loaded the original problem from {self.original_problem_manager.main_dir}")

    def load_conversion_data(self):
        with open(self.conversion_json_path, "r") as conversion_json_file:
            self.conversion_data = json.load(conversion_json_file)
        print(f"Loaded the problem conversion data from {self.conversion_json_path}")

    @property
    def conversion_json_path(self):
        return os.path.join(self.main_dir, "conversion.json")

    def generate_conversion_and_problem(self):
        self.problem, self.conversion_data = EfficientBTProblem.make_secretly_identical_instance(
            self.original_problem_manager.problem)
        print(f"Converted problem was generated")

    def generate_and_store_pRef(self):
        original_search_space_permutation = self.conversion_data["worker_permutation_dict"]
        search_space_permutation = [original_search_space_permutation[index]
                                    for index in range(len(original_search_space_permutation))]

        old_cardinalities = np.array(self.original_problem_manager.problem.search_space.cardinalities)
        new_search_space = SearchSpace(old_cardinalities[search_space_permutation])

        original_pRef = self.original_problem_manager.load_pRef()
        new_full_solution_matrix = original_pRef.full_solution_matrix[:, search_space_permutation]

        own_pRef = PRef(fitness_array=original_pRef.fitness_array,
                        search_space=new_search_space,
                        full_solution_matrix=new_full_solution_matrix)
        own_pRef.save(self.pRef_path)

        print(f"The converted pRef was obtained and stored in {self.pRef_path}")

    def generate_and_store_explanations(self,
                                        explanation_manager: ExplanationStorer):
        original_explanations = self.original_problem_manager.load_explanations()
        own_explanations = [explanation_manager.convert_explanation(expl, self.conversion_data)
                            for expl in original_explanations]

        # store the actual explanations
        for expl in own_explanations:
            self.store_explanation(expl, explanation_manager)

    def store_everything(self,
                         obtain_explanations_from_original_problem: bool):
        super().store_everything(generate_explanations_ex_novo=obtain_explanations_from_original_problem)

        with utils.open_and_make_directories(self.conversion_json_path) as file:
            json.dump(self.conversion_data, file, indent=4)

        print(f"Stored the conversion file into {self.conversion_json_path}")








def generate_for_second_problem(obtain_explanations_from_original_problem: bool):
    problem_manager = QuestionnaireDataForPermutedProblemGenerator(original_problem_folder=problem_A_path,
                                                                   own_problem_folder=problem_B_path)

    problem_manager.load_original_problem()
    problem_manager.generate_conversion_and_problem()
    problem_manager.store_everything(obtain_explanations_from_original_problem)





def big_bang():
    generate_for_first_problem(generate_explanations_ex_novo=True)
    generate_for_second_problem(obtain_explanations_from_original_problem=True)

    #generate_for_example_problem(generate_explanations_ex_novo=False)


def aftermath():
    problem_A_manager = QuestionnaireDataForProblemGenerator(main_dir=problem_A_path)
    problem_A_manager.load_problem()
    problem_B_manager = QuestionnaireDataForPermutedProblemGenerator(original_problem_folder=problem_A_path,
                                                                   own_problem_folder=problem_B_path)
    problem_B_manager.load_problem()

    finlay_mod = ("Finley", "B")
    amelia_mod = ("Amelia", "C")
    niamh_mod = ("Niamh", "A")
    question_1_options = [[finlay_mod],
                          [amelia_mod],
                          [niamh_mod],
                          [finlay_mod, amelia_mod],
                          [finlay_mod, niamh_mod],
                          [amelia_mod, niamh_mod]]
    problem_A_manager.store_answers(question_1_options, [])

    question_2_options = [
                            [("Ada", "A")],
                            [("Benjamin", "A")],
                            [("Sofia", "A")],
                            [("Eden", "A")],
                            [("Theo", "A")],
                          ]
    problem_B_manager.store_answers([], question_2_options)

#big_bang()


aftermath()
