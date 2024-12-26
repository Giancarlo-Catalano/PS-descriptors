import random
from typing import Any

import numpy as np
import pandas as pd

import utils
from BenchmarkProblems.BT.ReadFromFiles import get_dicts_from_RPD, make_roster_patterns_from_RPD, get_dicts_from_ED, \
    make_employees_from_ED, get_skills_dict
from BenchmarkProblems.BT.RotaPattern import get_workers_present_each_day_of_the_week, get_range_scores
from BenchmarkProblems.BT.Worker import Worker, WorkerVariables, Skill
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from resources.BT.names import names


class BTProblem(BenchmarkProblem):
    calendar_length: int
    workers: list[Worker]
    weights: list[float]
    all_skills: set[Skill]

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int,
                 weights=None):

        self.workers = workers
        self.calendar_length = calendar_length
        if weights is None:
            weights = [1, 1, 1, 1, 1, 10, 10]
        self.weights = weights

        assert (calendar_length % 7 == 0)

        variable_cardinalities = utils.join_lists(worker.get_variable_cardinalities(custom_starting_days=False)
                                                  for worker in self.workers)
        super().__init__(SearchSpace(variable_cardinalities))

        self.all_skills = set(skill for worker in self.workers
                              for skill in worker.available_skills)

    @classmethod
    def from_csv_files(cls, employee_data_file: str, employee_skills_file: str, rota_file: str, calendar_length: int):
        rpd_dicts = get_dicts_from_RPD(rota_file)
        rotas = make_roster_patterns_from_RPD(rpd_dicts)

        employee_dicts = get_dicts_from_ED(employee_data_file)
        employee_skills = get_skills_dict(employee_skills_file)
        employees = make_employees_from_ED(employee_dicts, employee_skills, rotas, names)
        return cls(employees, calendar_length)

    @classmethod
    def from_default_files(cls):
        # root = r"C:\Users\gac8\PycharmProjects\PS-PDF\resources\BT\MartinsInstance" + "\\"
        root = r"/Users/gian/PycharmProjects/PS-PDF/resources/BT/MartinsInstance/"  # for mac
        return cls.from_csv_files(employee_data_file=root + "employeeData.csv",
                                  employee_skills_file=root + "employeeSkillsData.csv",
                                  rota_file=root + "rosterPatternDaysData.csv",
                                  calendar_length=7 * 13)

    def to_json(self) -> dict:
        result = dict()
        result["calendar_length"] = self.calendar_length
        result["weights"] = self.weights
        result["workers"] = [worker.to_json() for worker in self.workers]
        return result

    def __repr__(self):
        return (f"BTProblem(amount_of_workers = {len(self.workers)}, "
                f"calendar_length={self.calendar_length}),"
                f"#skills={len(self.all_skills)})")

    def get_variables_from_fs(self, fs: FullSolution) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(fs.values), 1)

        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota = variable_list[0]
            return WorkerVariables(which_rota, starting_day=0)

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]

    def get_variables_from_ps(self, ps: PS) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(ps.values), 1)

        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota = variable_list[0]

            def from_value(value):
                if value == STAR:
                    return None
                else:
                    return value

            return WorkerVariables(from_value(which_rota), starting_day=0)

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]

    def get_range_score_from_wvs(self, wvs: list[WorkerVariables]) -> float:
        """assumes that all of the wvs's are valid, ie that none of the attributes are None"""
        rotas = [wv.effective_rota(worker, consider_starting_week=False)
                 for wv, worker in zip(wvs, self.workers)]

        rotas_by_skill = {skill: [rota for rota, worker in zip(rotas, self.workers)
                                  if skill in worker.available_skills]
                          for skill in self.all_skills}

        def range_score_for_rotas(input_rotas) -> float:
            calendar = get_workers_present_each_day_of_the_week(input_rotas, self.calendar_length)
            ranges = get_range_scores(calendar)
            return sum(range_score * weight for range_score, weight in zip(ranges, self.weights))

        return sum(range_score_for_rotas(rotas_by_skill[skill]) for skill in rotas_by_skill)

    def fitness_function(self, fs: FullSolution) -> float:
        wvs = self.get_variables_from_fs(fs)
        return -self.get_range_score_from_wvs(wvs)  # the minus is there because this is a minimisation problem

    def get_amount_of_first_choices(self, wvs: list[WorkerVariables]) -> int:
        return len([wv for wv in wvs if wv.which_rota == 0])

    def repr_ps(self, ps: PS) -> str:
        variables = self.get_variables_from_ps(ps)

        def repr_skills(available_skills: set[str]):
            if len(available_skills) > 0 and not next(iter(available_skills)).startswith("SKILL_"):
                return f"{available_skills}"
            integers = [int(skill.removeprefix("SKILL_")) for skill in available_skills]
            return f"{sorted(integers)}"

        return utils.indent("\n".join(
            f"{w.name} (Skills {repr_skills(w.available_skills)}, #rotas = {len(w.available_rotas)}): rota#{wv.which_rota}"
            for w, wv in zip(self.workers, variables)
            if wv.which_rota is not None))

    def details_of_solution(self, fs: FullSolution):
        wvs = self.get_variables_from_fs(fs)

        def get_relevant_rotas(skill):
            """assumes that all of the wvs's are valid, ie that none of the attributes are None"""
            return [wv.effective_rota(worker, consider_starting_week=False)
                    for wv, worker in zip(wvs, self.workers)
                    if skill in worker.available_skills]

        def mins_and_maxs(input_rotas):
            workers_per_weekday = get_workers_present_each_day_of_the_week(input_rotas, self.calendar_length)
            maxs = np.max(workers_per_weekday, axis=0)
            mins = np.min(workers_per_weekday, axis=0)
            return list(zip(mins, maxs))

        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ranges_for_skills = [[skill] + mins_and_maxs(get_relevant_rotas(skill))
                             for skill in self.all_skills]
        df = pd.DataFrame(ranges_for_skills, columns=["Skill"] + weekdays)
        return df

    @classmethod
    def from_json(cls, json_data: dict):
        workers = [Worker.from_json(worker_json) for worker_json in json_data["workers"]]
        calendar_length = json_data["calendar_length"]
        weights = json_data["weights"]
        return cls(workers=workers,
                   calendar_length=calendar_length,
                   weights=weights)

    @classmethod
    def make_secretly_identical_instance(cls, original_problem) -> (Any, dict):

        n = len(original_problem.workers)

        def get_name_conversion_dict() -> dict[str, str]:
            names_in_current = {worker.name for worker in original_problem.workers}
            assert (len(names_in_current) == n)
            available_names = set(names).difference(names_in_current)
            names_that_will_be_used = list(available_names)[:n]
            assert (len(names_in_current) == len(names_that_will_be_used))
            return {old_name: new_name for old_name, new_name in zip(names_in_current, names_that_will_be_used)}

        def get_worker_permutation_dict() -> dict[int, int]:
            new_indexes = list(range(len(original_problem.workers)))
            random.shuffle(new_indexes)
            return {old_index: new_index for old_index, new_index in enumerate(new_indexes)}

        def get_skill_permutation_dict() -> dict[str, str]:
            new_skills = list(original_problem.all_skills)
            random.shuffle(new_skills)
            return {old_skill: new_skill for old_skill, new_skill in zip(original_problem.all_skills, new_skills)}

        name_conversion_dict = get_name_conversion_dict()
        worker_permutation_dict = get_worker_permutation_dict()
        skill_permutation_dict = get_skill_permutation_dict()

        def get_new_worker(worker_index):
            original_worker = original_problem.workers[worker_permutation_dict[worker_index]]
            new_name = name_conversion_dict[original_worker.name]
            new_skills = {skill_permutation_dict[old_skill] for old_skill in original_worker.available_skills}
            new_worker = Worker(available_skills=new_skills,
                                name=new_name,
                                available_rotas=original_worker.available_rotas,
                                worker_id=original_worker.worker_id)  # this is never used so it shouldn't matter
            return new_worker

        new_workers = list(map(get_new_worker, range(n)))

        conversion_dict = {"name_conversion_dict": name_conversion_dict,
                           "worker_permutation_dict": worker_permutation_dict,
                           "skill_permutation_dict": skill_permutation_dict}

        new_problem = cls(workers=new_workers,
                                calendar_length=original_problem.calendar_length,
                                weights=original_problem.weights)

        return new_problem, conversion_dict
