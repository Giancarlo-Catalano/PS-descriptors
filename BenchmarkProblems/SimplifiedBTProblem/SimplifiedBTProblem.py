import itertools
import json
import os
import random
from typing import Optional

import numpy as np
import setuptools.errors

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from Explanation.PRefManager import PRefManager


class SimplifiedBTProblem(BenchmarkProblem):
    rotas: np.ndarray
    worker_names: list[str]
    skills: np.ndarray
    calendar_length: int
    qty_skills: int

    skill_names: list[str]

    worker_indices_for_each_skill = list[list[int]]

    original_indexes: Optional[list[int]]

    def __init__(self,
                 rotas: np.ndarray,
                 worker_names: list[str],
                 skills: np.ndarray,
                 skill_names: list[str],
                 original_indexes: Optional[list[int]] = None):
        self.rotas = rotas
        self.worker_names = worker_names
        self.skills = skills
        self.skill_names = skill_names

        variable_cardinality, self.calendar_length = rotas.shape
        qty_workers, self.qty_skills = skills.shape
        assert (qty_workers == len(worker_names))

        self.worker_indices_for_each_skill = [[index for index in range(qty_workers)
                                               if self.skills[index, skill_index]]
                                              for skill_index in range(self.qty_skills)]

        search_space = SearchSpace(variable_cardinality for worker in worker_names)
        self.original_indexes = original_indexes
        super().__init__(search_space)

    def differences_for_skill(self, fs: FullSolution, skill: int) -> int:
        indices_for_skill = self.worker_indices_for_each_skill[skill]
        counts_per_day = np.sum(self.rotas[fs.values[indices_for_skill]], axis=0)
        counts_per_day = counts_per_day.reshape(2, -1)
        return sum(np.abs(counts_per_day[0] - counts_per_day[1]))

    def worker_has_skill(self, worker_index: int, skill: int) -> bool:
        return self.skills[worker_index, skill]

    def skill_differences_matrix(self, ps: PS, skill: int) -> np.ndarray:
        rotas_of_active_relevant_workers = [self.rotas[ps.values[worker_index]]
                                            for worker_index in ps.get_fixed_variable_positions()
                                            if self.worker_has_skill(worker_index, skill)]
        counts_per_day = np.sum(rotas_of_active_relevant_workers, axis=0)
        counts_per_day = counts_per_day.reshape(2, -1)
        return np.abs(counts_per_day[0] - counts_per_day[1])

    def fitness_function(self, fs: FullSolution) -> float:
        return -float(sum(self.differences_for_skill(fs, skill) for skill in range(self.qty_skills)))

    @classmethod
    def from_json(cls, file_name: str):
        """
                {
                    rotas: [
                        "WWWW---WWWW---",
                        "WWW-W--WWW-W--",
                        "WWWW-W-WWWW-W-"
                    ],
                    skills: [
                       "X", "Y", "Z"
                    ],
                    workers: [
                        {"name": "Rhod",
                        "skills": ["X", "Y"]},
                        {"name": "Clara",
                        "skills": ["Z"]},
                    ]
                }
                """

        def read_rota_from_string(input_str: str):
            return np.array([c == 'W' for c in input_str])

        with open(file_name, "r") as file:
            data = json.load(file)

        rotas = np.array([read_rota_from_string(rota_str) for rota_str in data["rotas"]])
        skills = data["skills"]

        def read_skills_from_list(skill_list):
            return np.array([skill in skill_list for skill in skills])

        worker_names = [item["name"] for item in data["workers"]]
        skills_table = np.array([read_skills_from_list(worker["skills"]) for worker in data["workers"]])

        original_indexes = data.get("original_indexes", None)
        return cls(rotas=rotas, worker_names=worker_names, skills=skills_table, skill_names=data["skills"],
                   original_indexes=original_indexes)

    def to_dict(self):
        result = dict()

        def rota_to_str(rota: np.ndarray) -> str:
            return "".join("W" if cell else "-" for cell in rota)

        result["rotas"] = list(map(rota_to_str, self.rotas))
        result["skills"] = self.skill_names

        def convert_skill_row(skill_row: np.ndarray) -> list[str]:
            return [skill_name
                    for skill_name, is_present in zip(self.skill_names, skill_row)
                    if is_present]

        result["workers"] = [{"name": worker_name,
                              "skills": convert_skill_row(skill_row)}
                             for worker_name, skill_row in zip(self.worker_names, self.skills)]
        if self.original_indexes is not None:
            result["original_indexes"] = self.original_indexes
        return result

    def to_json(self, json_file_name: str):
        with utils.open_and_make_directories(json_file_name) as file:
            data = self.to_dict()
            json.dump(data, file, indent=4)

    def repr_fs(self, fs: FullSolution) -> str:
        return "".join(" " if value == STAR else utils.alphabet[value] for value in fs.values)

    def repr_ps(self, ps: PS) -> str:
        return ", ".join(f"{worker} = {utils.alphabet[value]}"
                         for worker, value in zip(self.worker_names, ps.values)
                         if value != STAR)

    def get_descriptors_of_ps(self, ps: PS) -> dict:
        workers_indexes = ps.get_fixed_variable_positions()
        rotas = ps.values[workers_indexes]

        skill_matrix = self.skills[workers_indexes]
        rotas_matrix = self.rotas[rotas]

        skillsets = [{skill for skill, is_used in zip(self.skill_names, row)} for row in skill_matrix]

        skill_counts = np.sum(skill_matrix, 0)
        total_qty_of_skills = np.sum(skill_counts > 0) # this might be wrong?

        def average_bivariate_distance(items, bivariate_aggregation, default) -> float:
            if len(items) == 0:
                return default

            return np.average([bivariate_aggregation(a, b)
                               for a, b in itertools.combinations(items, r=2)])

        def hamming_distance(a, b):
            return int(np.sum(a != b))

        skill_bivariate_distance = average_bivariate_distance(skill_matrix, hamming_distance, default=0)
        skill_properties = {"total_qty_of_skills": total_qty_of_skills,
                            "skill_bivariate_distance": skill_bivariate_distance,
                            "skills_shared_by_all": len(set.intersection(*skillsets)),
                            "average_qty_skills": np.average(np.sum(skill_matrix, 1)) if len(workers_indexes) > 0 else 0
                            }

        individual_skill_counts = {f"count_skill_{skill_name}": count
                                   for skill_name, count in zip(self.skill_names, skill_counts)}

        rota_average_bivariate_distance = average_bivariate_distance(rotas_matrix, hamming_distance, default=0)
        rota_properties = {"rota_average_bivariate_distance": rota_average_bivariate_distance}

        # days_worked = np.sum(rotas_matrix, axis=0)
        # differences = days_worked.reshape((2, -1))
        # differences = np.abs(differences[0] - differences[1])
        #
        # differences_by_weekday = {f"difference_{weekday}": difference
        #                           for weekday, difference in zip(utils.weekdays, differences)}

        skills_present = [index for index in range(self.qty_skills) if skill_counts[index] > 0]
        skill_differences_dict = {self.skill_names[skill]: self.skill_differences_matrix(ps, skill)
                                  for skill in skills_present}
        rota_differences = {f"diff_{weekday}_{skill_name}": diff_for_combinations
                            for skill_name, row_of_skill_diff in skill_differences_dict.items()
                            for weekday, diff_for_combinations in zip(utils.weekdays, row_of_skill_diff)
                            }

        return skill_properties | individual_skill_counts | rota_properties | rota_differences

    def get_permutated_pRef(self, original_pRef: PRef) -> PRef:
        return PRef(fitness_array=original_pRef.fitness_array,
                    full_solution_matrix=original_pRef.full_solution_matrix[:, self.original_indexes],
                    search_space=self.search_space)  # the search space remains the same because the cardinalities are all the same


    def repr_descriptors(self, descriptors: list[(str, float, float)]):

        diff_lines = []
        skill_count_lines = []
        other_lines = []

        for property_name, property_value, property_rank in descriptors:
            parts = property_name.split("_")
            if parts[0] == "diff" and property_value == 0.0:
                diff_lines.append(f"Balanced on {parts[1]}s on skill {parts[2]}")
            elif (parts[0], parts[1]) == ("count", "skill") and property_value > 0:
                skill_count_lines.append(f"{int(property_value)} people with skill \"{parts[2]}\"")
            elif property_name == "average_qty_skills":
                other_lines.append(f"{property_value:.2f} skills on average")
            elif property_name == "total_qty_of_skills":
                other_lines.append(f"{property_value:.2f} different skills present")
            elif property_name == "skill_bivariate_distance":
                other_lines.append("The skills are very similar" if property_rank < 0.5 else "The skills are very different")
            elif property_name == "rota_average_bivariate_distance":
                other_lines.append("The rotas are very similar" if property_rank < 0.5 else "The rotas are very different")
            else:
                other_lines.append(f"{property_name} = {property_value:.2f} <- rank {int(property_rank*100)}%")

        return "\n".join(itertools.chain(skill_count_lines, diff_lines, other_lines))



def test_simplified_problem():
    path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\BT\SimplifiedInstance\problem.json"
    problem = SimplifiedBTProblem.from_json(path)

    solutions_to_check = [FullSolution(0 for _ in range(problem.search_space.amount_of_parameters)),
                          FullSolution(1 for _ in range(problem.search_space.amount_of_parameters)),
                          FullSolution(2 for _ in range(problem.search_space.amount_of_parameters))]

    for solution in solutions_to_check:
        fitness = problem.fitness_function(solution)
        print(f"The solution {problem.repr_fs(solution)} has fitness {fitness}")

    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="GA",
                                     verbose=True)

    best_solution = pRef.get_best_solution()

    print(f"The best solution is {problem.repr_fs(best_solution)}, it has fitness {best_solution.fitness}")


# test_simplified_problem()


def make_shuffled_instance():
    folder_a = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\A"
    folder_b = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\B"

    problem_a_file = os.path.join(folder_a, "problem.json")

    problem_a = SimplifiedBTProblem.from_json(problem_a_file)

    # first, we generate the new rotas: shuffle the days and the rotas' order

    weekdays_reassignment = utils.shuffled(range(7))

    def get_shuffled_rota(rota: np.ndarray) -> np.ndarray:
        # organise in weeks, shuffle the columns, and then flatten
        new_rota = rota.reshape((-1, 7))
        new_rota = new_rota[:, weekdays_reassignment]
        return new_rota.ravel()

    new_rotas = np.array(utils.shuffled(map(get_shuffled_rota, problem_a.rotas)))

    # then we shuffle the workers and give them new names
    problem_b_names = ["Alice", "Brandon", "Clara", "Dominic", "Eleanor", "Fiona", "Gabriel", "Hazel", "Ivy", "Joshua",
                       "Kevin", "Lucas", "Matilda", "Nathan", "Oscar", "Phoebe", "Quentin", "Rose", "Sophia"]
    worker_qty = len(problem_a.worker_names)
    assert len(problem_b_names) >= worker_qty

    worker_indexes_reassignment = utils.shuffled(range(worker_qty))
    new_skill_names = ["security", "management", "sales", "cloud", "database"]

    new_skill_matrix = problem_a.skills[worker_indexes_reassignment]

    problem_b = SimplifiedBTProblem(rotas=new_rotas,
                                    worker_names=problem_b_names,
                                    skills=new_skill_matrix,
                                    skill_names=new_skill_names,
                                    original_indexes=worker_indexes_reassignment)

    problem_b_destination = os.path.join(folder_b, "problem.json")
    problem_b.to_json(problem_b_destination)


#make_shuffled_instance()
