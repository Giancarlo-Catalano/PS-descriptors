import itertools
import json

import numpy as np
import setuptools.errors

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
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

    def __init__(self,
                 rotas: np.ndarray,
                 worker_names: list[str],
                 skills: np.ndarray,
                 skill_names: list[str]):
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
        super().__init__(search_space)

    def differences_for_skill(self, fs: FullSolution, skill: int) -> int:
        indices_for_skill = self.worker_indices_for_each_skill[skill]
        counts_per_day = np.sum(self.rotas[fs.values[indices_for_skill]], axis=0)
        counts_per_day = counts_per_day.reshape(2, -1)
        return sum(np.abs(counts_per_day[0] - counts_per_day[1]))

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

        return cls(rotas=rotas, worker_names=worker_names, skills=skills_table, skill_names=data["skills"])

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
        total_qty_of_skills = np.sum(skill_counts > 0)

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

        days_worked = np.sum(rotas_matrix, axis=0)
        differences = days_worked.reshape((2, -1))
        differences = np.abs(differences[0] - differences[1])

        differences_by_weekday = {f"difference_{weekday}": difference
                                  for weekday, difference in zip(utils.weekdays, differences)}

        return skill_properties | individual_skill_counts | rota_properties | differences_by_weekday


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
