import random
from typing import Optional

import numpy as np

import utils
from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FSEvaluator import FSEvaluator
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU, WilcoxonTest, \
    WilcoxonNearOptima, get_hypothesis_string
from LCS.DifferenceExplainer.DescriptorsManager import DescriptorsManager


class BTProblemPrettyPrinter:
    problem: EfficientBTProblem
    descriptor_manager: Optional[DescriptorsManager]

    all_rotas_list: list[RotaPattern]
    all_skills_list: list[str]

    skill_emoji_dict: Optional[dict[str, str]]

    def __init__(self,
                 problem: EfficientBTProblem,
                 skill_emoji_dict: Optional[dict[str, str]],
                 descriptor_manager: Optional[DescriptorsManager] = None
                 ):
        self.problem = problem

        self.descriptor_manager = descriptor_manager

        self.all_rotas_list = self.get_all_rotas_list(self.problem)
        self.all_skills_list = sorted(list(self.problem.all_skills))
        self.skill_emoji_dict = skill_emoji_dict

    @classmethod
    def get_all_rotas_list(cls, problem: EfficientBTProblem) -> list[RotaPattern]:
        # note that two rotas which are equivalent might have different hash values,
        # so simply making a set of them is not guaranteed to remove duplicates
        # if we force the patterns to be in their minimal form, it should work!

        unique_rotas = list({cls.simplify_rota(rota)
                     for worker in problem.workers
                     for rota in worker.available_rotas})

        unique_rotas.sort(key=hash)
        seed = hash(worker.name for worker in problem.workers)
        random.seed(seed)
        return unique_rotas

    @classmethod
    def simplify_rota(cls, rota: RotaPattern) -> RotaPattern:
        """ The pattern [WWW---- WWW----] can be simplified into [WWW----] """

        """There's absolutely a better way to do this but I'm worried of not handling edge cases, so here's a plain approach"""

        def cut_rota(rota: RotaPattern, amount_of_weeks: int) -> RotaPattern:
            days = rota.days[:(7 * amount_of_weeks)]
            return RotaPattern(workweek_length=7, days=days)

        original_rota_weeks = len(rota.days) // 7

        replacement_rota = rota
        for size_of_rota in range(1, original_rota_weeks):
            smaller_rota = cut_rota(rota, size_of_rota)
            if smaller_rota == rota:
                replacement_rota = smaller_rota
                break

        return replacement_rota


    def repr_skill(self, skill: str) -> str:
        if self.skill_emoji_dict is None:
            return skill
        else:
            return self.skill_emoji_dict[skill]

    def get_index_of_rota(self, rota: RotaPattern) -> int:
        return self.all_rotas_list.index(rota)

    def repr_rota_choice(self, rota_index: int) -> str:
        return utils.alphabet[rota_index]

    def repr_rota_index(self, rota: RotaPattern) -> str:
        index = self.all_rotas_list.index(rota)
        return f"ROTA {index + 1}"

    def repr_rota(self, rota: RotaPattern) -> str:
        return "\t".join(f"{day}" for day in rota.days)

    def repr_extended_rota(self, rota: RotaPattern) -> str:
        new_days = []
        while len(new_days) < self.problem.calendar_length:
            new_days.extend(rota.days)

        new_days = new_days[:self.problem.calendar_length]
        new_rota = RotaPattern(workweek_length=rota.workweek_length, days=new_days)
        return self.repr_rota(new_rota)

    def repr_skillset(self, skillset: set[str]) -> str:
        return "\t".join((self.repr_skill(skill) if skill in skillset else "") for skill in self.all_skills_list)

    def repr_worker(self, worker: Worker) -> str:
        skills_str = self.repr_skillset(worker.available_skills)
        rotas_str = "\t".join(self.repr_rota_index(rota) for rota in worker.available_rotas)
        return "\t".join([worker.name, skills_str, rotas_str])

    def get_sorted_workers(self) -> list[Worker]:
        return sorted(self.problem.workers, key=lambda w: w.name)
    def repr_problem_workers(self) -> str:
        return "\n".join(map(self.repr_worker, self.get_sorted_workers()))

    def repr_problem_rotas(self) -> str:
        return "\n".join("\t".join([self.repr_rota_index(rota), self.repr_rota(rota)])
                         for rota in self.all_rotas_list)

    def repr_partial_solution(self, ps: PS) -> str:
        workers_and_choices = [(worker, choice)
                               for worker, choice in zip(self.problem.workers, ps.values)
                               if choice != STAR]

        workers_and_choices = sorted(workers_and_choices, key = lambda wc : wc[0].name)

        def repr_assigned_worker(worker: Worker, choice: int) -> str:
            actual_rota = worker.available_rotas[choice]
            skillset_str = self.repr_skillset(worker.available_skills)
            rota_choice_label = self.repr_rota_choice(choice)
            rota_index_label = self.repr_rota_index(actual_rota)
            actual_rota_str = self.repr_extended_rota(actual_rota)

            return "\t".join([worker.name, rota_choice_label, rota_index_label, skillset_str, actual_rota_str])

        return "\n".join([repr_assigned_worker(w, c) for w, c in workers_and_choices])

    def repr_extra_information_for_partial_solution(self,
                                                    ps: PS,
                                                    hypothesis_tester: WilcoxonTest,
                                                    near_optima_hypothesis_tester: WilcoxonNearOptima) -> str:
        """ The pRef is used to carry out some hypothesis testing on near_optima solutions"""
        calendar = self.get_calendar_counts_for_ps(ps)
        calendar_string = self.repr_skill_calendar(calendar)
        penalties_strings = self.get_penalties_string(calendar)
        hypothesis_string = get_hypothesis_string(ps, hypothesis_tester, near_optima_hypothesis_tester)
        return "\n\n".join([calendar_string, penalties_strings, hypothesis_string])

    def repr_extra_information_for_full_solution(self, fs: FullSolution) -> str:
        ps = PS.from_FS(fs)
        calendar = self.get_calendar_counts_for_ps(ps)
        calendar_string = self.repr_skill_calendar(calendar)
        penalties_strings = self.get_penalties_string(calendar)
        return "\n".join([calendar_string, penalties_strings])

    def repr_full_solution(self, fs: FullSolution) -> str:
        return self.repr_partial_solution(PS.from_FS(fs))

    def get_calendar_counts_for_ps(self, ps: PS) -> dict:
        present_rotas_and_skills = [
            (self.problem.extended_patterns[index][choice], self.problem.workers[index].available_skills)
            for index, choice in enumerate(ps.values) if choice != STAR]

        def get_calendar_for_skill(skill: str) -> np.ndarray:
            relevant_patterns = [pattern for pattern, skillset in present_rotas_and_skills if skill in skillset]
            if len(relevant_patterns) == 0:
                return np.zeros(shape=self.problem.calendar_length, dtype=int)
            return np.sum(relevant_patterns, axis=0)

        return {skill: get_calendar_for_skill(skill) for skill in self.all_skills_list}


    def repr_skill_calendar(self, skill_calendar: dict) -> str:
        def repr_for_skill(skill: str) -> str:
            return "\t".join([self.repr_skill(skill)] + [f"{x}" for x in skill_calendar[skill]])

        return "\n".join(repr_for_skill(skill) for skill in self.all_skills_list)

    def get_penalties_string(self, calendar: dict[str, np.ndarray]) -> str:
        def get_penalty_string(counts_of_workers: list[int]) -> str:
            least = min(counts_of_workers)
            most = max(counts_of_workers)

            fitness = 1.0 if least == 0 else ((most - least) / most) ** 2
            return f"\"max = {most},\nmin = {least},\n p = {fitness:.2f}\""

        def repr_for_skill(skill: str) -> str:
            counts = calendar[skill]
            counts_by_weekday = counts.reshape((-1, 7))
            counts_by_weekday = [list(counts_by_weekday[:, col]) for col in range(7)]
            penalty_strings = list(map(get_penalty_string, counts_by_weekday))
            return "\t".join([self.repr_skill(skill)] + penalty_strings)

        return "\n".join(repr_for_skill(skill) for skill in self.all_skills_list)

    def repr_difference_between_solutions(self,
                                          main_solution: FullSolution,
                                          background_solution: FullSolution) -> str:
        n = self.problem.search_space.amount_of_parameters
        differences = [(index, value_in_a, value_in_b)
                       for index, value_in_a, value_in_b
                       in zip(range(n), main_solution.values, background_solution.values)
                       if value_in_a != value_in_b]

        def repr_difference(difference: (int, int, int)) -> str:
            index, value_in_a, value_in_b = difference
            worker_name = self.problem.workers[index].name
            return (f"{worker_name}, "
                    f"in main has choice {self.repr_rota_choice(value_in_a)},"
                    f"in background has choice {self.repr_rota_choice(value_in_b)}")

        return "\n".join(map(repr_difference, differences))

    def get_worker_name(self, worker_index: int) -> str:
        return self.problem.workers[worker_index].name

    def get_value_as_rota_index(self, worker_index: int, chosen_rota: int) -> str:
        rota = self.problem.workers[worker_index].available_rotas[chosen_rota]
        return self.repr_rota_index(rota)



    def repr_day(self, day_index: int) -> str:
        week = (day_index // 7) + 1
        weekday = utils.weekdays[day_index % 7]
        return f"Week {week}, {weekday}"
