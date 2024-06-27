import itertools
import math
from typing import TypeAlias

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_range_scores
from BenchmarkProblems.BT.Worker import Worker
from FirstPaper.PS import PS, STAR
from FirstPaper.custom_types import JSON

ExtendedPattern: TypeAlias = np.ndarray


def rota_to_extended_pattern(rota: RotaPattern, calendar_length: int) -> ExtendedPattern:
    pattern = np.array([day.working for day in rota.days], dtype=int)
    if len(pattern) >= calendar_length:
        return pattern[:calendar_length]

    return np.tile(pattern, math.ceil(calendar_length / len(pattern)))[:calendar_length]


def get_rotated_by_starting_week(full_pattern: ExtendedPattern, starting_week: int) -> ExtendedPattern:
    return np.roll(full_pattern, -starting_week)


def convert_worker_to_just_options(worker: Worker, calendar_length: int) -> np.ndarray:
    return np.array([rota_to_extended_pattern(rota, calendar_length)
                     for rota in worker.available_rotas])


FullPatternOptions: TypeAlias = np.ndarray
DayRange: TypeAlias = float
WeekRanges: TypeAlias = np.ndarray


class CohortMember:
    worker: Worker
    chosen_rota_index: int
    chosen_rota_entended: ExtendedPattern


    def __init__(self,
                 worker: Worker,
                 rota_index: int,
                 calendar_length: int):
        self.worker = worker
        self.chosen_rota_index = rota_index
        self.chosen_rota_extended = rota_to_extended_pattern(rota=worker.available_rotas[rota_index],
                                                             calendar_length=calendar_length)

    def to_json(self) -> JSON:
        return {"worker": self.worker.to_json(),
                "chosen_rota": int(self.chosen_rota_index)}

    @classmethod
    def from_json(cls, element: JSON):
        chosen_rota = int(element["chosen_rota"])
        calendar_length = 91  # TODO fetch this from somewhere else
        worker = Worker.from_json(element["worker"])
        return cls(worker=worker, rota_index=chosen_rota, calendar_length=calendar_length)


    def get_amount_of_skills(self) -> int:
        return len(self.worker.available_skills)

    def get_mean_weekly_working_days(self) -> int:
        total_working_days = np.sum(self.chosen_rota_extended)
        total_weeks = len(self.chosen_rota_extended) // 7
        return total_working_days / total_weeks


    def get_amount_of_choices(self) -> int:
        return len(self.worker.available_rotas)



    def get_proportion_of_working_saturdays(self) -> float:
        return np.average(self.chosen_rota_extended.reshape((-1, 7))[:, 5])


Cohort: TypeAlias = list[CohortMember]


def ps_to_cohort(problem: BTProblem, ps: PS) -> Cohort:
    def fixed_var_to_cohort_member(var: int, val: int) -> CohortMember:
        worker = problem.workers[var]
        return CohortMember(worker, val, calendar_length=problem.calendar_length)
    return [fixed_var_to_cohort_member(var, val)
            for var, val in enumerate(ps.values)
            if val != STAR]


def cohort_to_json(cohort: Cohort) -> JSON:
    return [member.to_json() for member in cohort]


def get_amount_of_shared_skills(cohort: Cohort) -> int:
    if len(cohort) == 0:
        return 0

    skillsets = [component.worker.available_skills for component in cohort]
    common_to_all = set.intersection(*skillsets)
    return len(common_to_all)


def get_skill_variation(cohort: Cohort) -> float:
    all_skills = set(skill for component in cohort
                          for skill in component.worker.available_skills)
    sum_of_available_skills = sum(len(component.worker.available_skills) for component in cohort)
    return len(all_skills) / sum_of_available_skills


def get_skill_coverage(cohort: Cohort) -> float:
    all_skills = set(skill for component in cohort
                     for skill in component.worker.available_skills)
    return len(all_skills)


def get_hamming_distances(cohort: Cohort) -> list[int]:

    def hamming_distance(component_a: CohortMember, component_b: CohortMember) -> int:
        rota_a = component_a.chosen_rota_extended
        rota_b = component_b.chosen_rota_extended

        return int(np.sum(rota_a != rota_b))

    if len(cohort) == 2:  # makes my life a lot easier for data analysis
        distance = hamming_distance(cohort[0], cohort[1])
        return [distance, distance]

    return [hamming_distance(a, b)
               for a, b in itertools.combinations(cohort, 2)]


def get_ranges_in_weekdays(cohort: Cohort, use_faulty_fitness_function = False) -> np.ndarray:
    total_pattern: np.ndarray = np.array(sum(member.chosen_rota_extended for member in cohort))
    total_pattern = total_pattern.reshape((-1, 7))
    return get_range_scores(total_pattern, use_faulty_fitness_function)


def get_mins_and_maxs_for_weekdays(cohort: Cohort) -> list[(int, int)]:
    total_pattern: np.ndarray = np.array(sum(member.chosen_rota_extended for member in cohort))
    total_pattern = total_pattern.reshape((-1, 7))
    mins = np.min(total_pattern, axis=0)
    maxs = np.max(total_pattern, axis=0)
    return list(zip(mins, maxs))


def get_coverage(cohort: Cohort) -> float:
    """returns the proportion of days in the calendar with at least one worker"""
    total_pattern = np.array(sum(member.chosen_rota_extended for member in cohort))
    total_pattern = np.minimum(total_pattern, 1)
    return np.average(total_pattern)   # happens to be equal to quantity_working_days / quantity_days


def get_amount_of_covered_weekends(cohort: Cohort) -> (float, float):
    covered_days = sum(member.chosen_rota_extended for member in cohort)
    covered_days = np.array(covered_days, dtype = bool).reshape((-1, 7))
    covered_saturdays = np.sum(covered_days[:, 5], dtype=float)
    covered_sundays = np.sum(covered_days[:, 6], dtype=float)


    return float(covered_saturdays), float(covered_sundays)
