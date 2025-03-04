import itertools
import json
import os

import utils
from BenchmarkProblems.SimplifiedBTProblem.SimplifiedBTProblem import SimplifiedBTProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from UserStudy.problem_and_explanations_script import get_problem_path, get_pRef_path, instance_ca_path


def modifications_to_dict(modifications: list[list[(str, str)]]) -> list[list[dict]]:
    def single_pair_to_dict(modif_item: (str, str)) -> dict:
        return {"person": modif_item[0],
                "rota": modif_item[1]}

    return [[single_pair_to_dict(pair) for pair in modif] for modif in modifications]


def modifications_from_dict(modif_dict: list[list[dict]]) -> list[list[(str, str)]]:
    def read_pair(pair_dict) -> (str, str):
        return (pair_dict["person"], pair_dict["rota"])

    return [[read_pair(item) for item in modif_list]
            for modif_list in modif_dict]


def write_modifications_to_file(modifications_task_1, modifications_task_2, file_path: str):
    data = {
        "task_1": modifications_to_dict(modifications_task_1),
        "task_2": modifications_to_dict(modifications_task_2),
    }

    with utils.open_and_make_directories(file_path) as file:
        json.dump(data, file, indent=4)
    print(f"Saved the modifications to the file {file_path}")


def get_task_path_for_instance(instance_path: str) -> str:
    return os.path.join(instance_path, "tasks.json")


def load_task_modifications_from_path(task_path: str) -> (list[list[(str, str)]], list[list[(str, str)]]):
    with open(task_path, "r") as file:
        data = json.load(file)

    task_1_modifs = modifications_from_dict(data["task_1"])
    task_2_modifs = modifications_from_dict(data["task_2"])
    return task_1_modifs, task_2_modifs


instance_cb_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\Constructed_B"


def write_tasks_for_instance_cb():
    task_1_modifications = [
        [("Brandon", "D"), ("Kevin", "C")],
        [("Brandon", "D")],
        [("Phoebe", "A")],
        [("Phoebe", "A"), ("Brandon", "D")]
    ]

    to_combine = [("Alice", "B"), ("Lucas", "B"), ("Phoebe", "A"),
                    ("Brandon", "D"), ("Kevin", "C")]
    task_2_modifications = [(x, y)
                            for x, y in itertools.combinations(to_combine, r=2)]

    instance_cb_modif_path = get_task_path_for_instance(instance_path=instance_cb_path)
    write_modifications_to_file(task_1_modifications, task_2_modifications, instance_cb_modif_path)


#write_tasks_for_instance_cb()


def check_answers(instance_path):
    problem_path = get_problem_path(instance_path)
    pRef_path = get_pRef_path(instance_path)

    problem = SimplifiedBTProblem.from_json(problem_path)
    pRef = PRef.load(pRef_path)

    def with_single_modification(solution: FullSolution, name_and_rota) -> FullSolution:
        name, rota = name_and_rota
        worker_index = problem.worker_names.index(name)
        rota_index = utils.alphabet.index(rota)

        if solution.values[worker_index] == rota_index:
            print(f"Warning!: the worker {name} was already on rota {rota}")

        return solution.with_different_value(worker_index, rota_index)

    def with_modifications(solution: FullSolution, modifications) -> EvaluatedFS:
        new_solution = solution.copy()
        for item in modifications:
            new_solution = with_single_modification(new_solution, item)
        return EvaluatedFS(new_solution, fitness=problem.fitness_function(new_solution))

    best_solution = pRef.get_best_solution()
    print(
        f"The best solution has fitness {best_solution.fitness}, and its {problem.repr_ps(PS.from_FS(best_solution))}")

    task_path = get_task_path_for_instance(instance_path)
    task_1_modifications, task_2_modifications = load_task_modifications_from_path(task_path)

    def show_modification_ordered(mods):
        modified_solutions = [(modifs, with_modifications(best_solution, modifs))
                              for modifs in mods]
        modified_solutions.sort(key=lambda x: x[1].fitness, reverse=True)

        for modif, solution in modified_solutions:
            print(f"With modification {modif}, the fitness is {solution.fitness}")

    print("For task 1")
    show_modification_ordered(task_1_modifications)

    print("For task 2")
    show_modification_ordered(task_2_modifications)


check_answers(instance_ca_path)
check_answers(instance_cb_path)
