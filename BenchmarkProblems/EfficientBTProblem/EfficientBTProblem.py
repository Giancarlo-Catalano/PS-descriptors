import copy
import itertools
import math
import random
from typing import TypeAlias, Optional, Any

import numpy as np

import utils
from BenchmarkProblems import InverseGraphColouringProblem
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_range_scores, WorkDay
from BenchmarkProblems.BT.Worker import Worker, Skill
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.custom_types import JSON

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


def get_ranges_in_weekdays(cohort: Cohort, use_faulty_fitness_function=False) -> np.ndarray:
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
    return np.average(total_pattern)  # happens to be equal to quantity_working_days / quantity_days


def get_qty_of_covered_weekends(cohort: Cohort) -> list[float]:
    covered_days = sum(member.chosen_rota_extended for member in cohort)
    covered_days = np.array(covered_days, dtype=bool).reshape((-1, 7))

    covered_weekdays = np.sum(covered_days, axis=1, dtype=float)
    return list(covered_weekdays)


class EfficientBTProblem(BTProblem):
    extended_patterns: list[FullPatternOptions]
    workers_by_skills: dict  # Skill -> set[worker index]
    use_faulty_fitness_function: bool
    rota_preference_weight: float

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int,
                 use_faulty_fitness_function: bool = False,
                 weights: list[float] = None,
                 rota_preference_weight: float = 0.001):
        super().__init__(workers, calendar_length, weights=weights)
        self.extended_patterns = [convert_worker_to_just_options(worker, calendar_length)
                                  for worker in workers]
        self.workers_by_skills = {skill: {index for index, worker in enumerate(self.workers)
                                          if skill in worker.available_skills}
                                  for skill in self.all_skills}
        self.use_faulty_fitness_function = use_faulty_fitness_function
        self.rota_preference_weight = rota_preference_weight

    def get_ranges_for_weekdays_for_skill(self, chosen_patterns: list[ExtendedPattern],
                                          skill: Skill) -> WeekRanges:
        indexes = self.workers_by_skills[skill]
        summed_patterns: ExtendedPattern = np.sum([chosen_patterns[index] for index in indexes],
                                                  axis=0)
        summed_patterns = summed_patterns.reshape((-1, 7))
        return get_range_scores(summed_patterns, self.use_faulty_fitness_function)

    def aggregate_range_scores(self, range_scores: WeekRanges) -> float:
        return float(np.sum(day_range * weight for day_range, weight in zip(range_scores, self.weights)))

    def get_chosen_patterns_from_fs(self, fs: FullSolution) -> list[ExtendedPattern]:
        return [options[which] for options, which in zip(self.extended_patterns, fs.values)]

    def fitness_function(self, fs: FullSolution) -> float:
        chosen_patterns = self.get_chosen_patterns_from_fs(fs)
        quantity_of_unliked_rotas = np.sum(fs.values != 0)

        def score_for_skill(skill) -> float:
            ranges = self.get_ranges_for_weekdays_for_skill(chosen_patterns, skill)
            return self.aggregate_range_scores(ranges)

        rota_score = np.sum([score_for_skill(skill) for skill in self.all_skills])
        preference_score = self.rota_preference_weight * quantity_of_unliked_rotas
        return -(rota_score + preference_score)  # to convert it to a maximisation task

    def get_descriptors_of_ps(self, ps: PS) -> dict:
        cohort = ps_to_cohort(self, ps)

        choice_amounts = [member.get_amount_of_choices() for member in cohort]
        weekly_working_days = [member.get_mean_weekly_working_days() for member in cohort]
        rota_differences = get_hamming_distances(cohort)
        working_saturday_proportions = [member.get_proportion_of_working_saturdays() for member in cohort]
        skill_quantities = [len(member.worker.available_skills) for member in cohort]

        # the local fitness is not a good metric to use
        local_fitness = np.average(get_ranges_in_weekdays(cohort, self.use_faulty_fitness_function))
        quantity_of_fav_rotas = len([worker for worker in cohort if worker.chosen_rota_index == 0])

        mean_RCQ, mean_error_RCQ = utils.get_mean_and_mean_error(choice_amounts)
        mean_WWD, mean_error_WWD = utils.get_mean_and_mean_error(weekly_working_days)
        mean_RD, mean_error_RD = utils.get_mean_and_mean_error(rota_differences)
        # mean_WSP, mean_error_WSP = utils.get_mean_and_mean_error(working_saturday_proportions)
        mean_SQ, mean_error_SQ = utils.get_mean_and_mean_error(skill_quantities)

        coverage_weekdays = get_qty_of_covered_weekends(cohort)

        coverage = get_coverage(cohort)
        skill_diversity = get_skill_variation(cohort)
        skill_coverage = get_skill_coverage(cohort)

        size = ps.fixed_count()

        return {  # "mean_RCQ" : mean_RCQ,
            # "mean_error_RCQ" : mean_error_RCQ,
            # "mean_WWD" : mean_WWD,
            # "mean_error_WWD" : mean_error_WWD,
            "mean_RD": mean_RD,
            # "mean_error_RD" : mean_error_RD,
            "covered_monday": coverage_weekdays[0],
            "covered_tuesday": coverage_weekdays[1],
            "covered_wednesday": coverage_weekdays[2],
            "covered_thursday": coverage_weekdays[3],
            "covered_friday": coverage_weekdays[4],
            "covered_saturday": coverage_weekdays[5],
            "covered_sunday": coverage_weekdays[6],
            # "mean_error_WSP" : mean_error_WSP,
            "mean_SQ": mean_SQ,
            # "mean_error_SQ": mean_error_SQ,
            # "skill_diversity": skill_diversity,
            "skill_coverage": skill_coverage,
            # "day_coverage": coverage,
            "quantity_of_fav_rotas": quantity_of_fav_rotas
        }

    def repr_extra_ps_info(self, ps: PS):
        cohort = ps_to_cohort(self, ps)
        mins_maxs = get_mins_and_maxs_for_weekdays(cohort)
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"The ranges are " + (", ".join(f"{weekday}:{min_max}" for weekday, min_max in zip(weekdays, mins_maxs)))

    def repr_property(self, property_name: str, property_value: float, rank: (float, float), ps: PS):
        # lower_rank, upper_rank = property_rank_range
        is_low = rank < 0.5
        rank_str = f"(rank = {int(rank * 100)}%)"  # "~ {int(property_rank_range[1]*100)}%)"

        cohort = ps_to_cohort(self, ps)

        if property_name == "mean_RCQ":
            rota_choice_quantities = [member.get_amount_of_choices() for member in cohort]
            return (f"The workers have {'FEW' if is_low else 'MANY'} rota choices: {rota_choice_quantities} "
                    f"(mean = {np.average(rota_choice_quantities):.2f}, {rank_str})")
        # elif property_name == "mean_error_RCQ":
        #     rota_choice_quantities = [member.get_amount_of_choices() for member in cohort]
        #     return (f"The workers have {'THE SAME' if is_low else 'DIFFERENT'} "
        #             f"amounts of rota choices: {rota_choice_quantities} rank = {rank_str})")
        elif property_name == "mean_WWD":
            working_week_days = [member.get_mean_weekly_working_days() for member in cohort]
            return (f"The selected rotas have {'FEW' if is_low else 'MANY'} working days "
                    f"(avg per week, per worker: {working_week_days}, {rank_str})")
        # elif property_name == "mean_error_WWD":
        #     working_week_days = [member.get_mean_weekly_working_days() for member in cohort]
        #     return (f"The selected rotas have {'FEW' if is_low else 'MANY'} working days  "
        #             f"average per week, per worker: {working_week_days} rank = {rank_str})")
        elif property_name == "mean_RD":
            average_difference = np.average(get_hamming_distances(cohort))
            return (f"The selected rotas are {'SIMILAR' if is_low else 'DIFFERENT'} "
                    f"(avg diff = {average_difference:.2f}), {rank_str})")
        elif property_name == "covered_sats":
            return (f"The selected rotas cover {'few' if is_low else 'many'} "
                    f"Saturdays: {int(property_value)} are covered, {rank_str}")
        elif property_name == "covered_suns":
            working_saturday_proportions = [member.get_proportion_of_working_saturdays() for member in cohort]
            covered_saturdays = int(
                np.average(working_saturday_proportions) * len(cohort) * (self.calendar_length // 7))
            return (f"The selected rotas cover {'few' if is_low else 'many'} "
                    f"Sundays: {int(property_value)} are covered, {rank_str}")
        elif property_name == "mean_SQ":
            return (f"The workers have {'FEW' if is_low else 'MANY'} skills, {rank_str}")
        elif property_name == "skill_diversity":
            return (f"The skills are {'SIMILAR' if is_low else 'DIVERSE'}, {rank_str}")
        elif property_name == "skill_coverage":
            return (f"The skills cover a {'NARROW' if is_low else 'WIDE'} range, {rank_str}")
        elif property_name == "quantity_of_fav_rotas":
            return (f"{'FEW' if is_low else 'MANY'} workers got their preferred rota, {rank_str}")
        elif property_name == "delta":
            return (f"The delta fitness = {property_value:.2f}, which has percentile {rank_str}")
        elif property_name.startswith("covered_"):
            weekday = property_name.split("_")[1]
            return f"On {weekday}, the coverage is {'LOW' if is_low else 'HIGH'}, {rank_str}"
            raise ValueError(f"Did not recognise the property {property_name} in EfficientBTProblem")

    @classmethod
    def from_Graph_Colouring(cls, gc: GraphColouring):

        working_week = [WorkDay.working_day(900, 1700) for _ in range(7)]
        not_working_week = [WorkDay.not_working() for _ in range(7)]

        def make_rota_option_for_colour(colour_index: int) -> RotaPattern:
            weeks_to_use = [not_working_week if colour_index == i else working_week
                            for i in range(gc.amount_of_colours)]
            all_days = utils.flatten(weeks_to_use)
            return RotaPattern(7, all_days)

        def make_worker(node_number: int):
            rota_options = [make_rota_option_for_colour(c) for c in range(gc.amount_of_colours)]
            return Worker(available_skills=set(),
                          available_rotas=rota_options,
                          name=f"Node_{node_number}",
                          worker_id=f"Node_{node_number}")

        workers = [make_worker(node_index) for node_index in range(gc.amount_of_nodes)]

        for skill_number, connection in enumerate(gc.connections):
            skill_str = f"SKILL_{skill_number}"
            start, end = connection
            workers[start].available_skills.add(skill_str)
            workers[end].available_skills.add(skill_str)

        return EfficientBTProblem(workers,
                                  calendar_length=7 * gc.amount_of_colours,
                                  rota_preference_weight=0,
                                  weights=[1 for _ in range(7)])

    @classmethod
    def from_RoyalRoad(cls, rr: RoyalRoad):
        amount_of_days = 7 * rr.clique_size
        master_rota_days = [WorkDay.working_day(900, 1700)
                            if i < 7 else WorkDay.not_working() for i in range(amount_of_days)]
        no_working_days = RotaPattern(7, [WorkDay.not_working() for i in range(amount_of_days)])

        def make_worker_in_clique(which_worker: int, which_clique: int) -> Worker:
            rotas = [no_working_days,
                     RotaPattern(7, utils.cycle(master_rota_days, 7 * which_worker))]
            name = f"W{which_clique}_{which_worker}"
            id = name
            skills = {f"SKILL_{which_clique}"}
            return Worker(available_skills=skills, available_rotas=rotas, name=name, worker_id=id)

        workers = [make_worker_in_clique(which_worker, which_clique)
                   for which_clique in range(rr.amount_of_cliques)
                   for which_worker in range(rr.clique_size)]

        return EfficientBTProblem(workers=workers, calendar_length=amount_of_days,
                                  weights=[1 for _ in range(7)], rota_preference_weight=0)

    def get_readable_property_name(self, property: str) -> str:
        match property:
            case "mean_RCQ":
                return "rota choice amount"
            case "mean_error_RCQ":
                return "difference in rota choice amount"
            case "mean_RD":
                return "difference between rotas"
            case "mean_WSP":
                return "saturday coverage"
            case "mean_SQ":
                return "amount of skills"
            case "skill_diversity":
                return "similarity of skills"
            case "skill_coverage":
                return "collective coverage of skills"
            case "quantity_of_fav_rotas":
                return "preferred rota usage"
            case _:
                return property

    def print_stats_of_pss(self, pss: list[PS], full_solutions: list[EvaluatedFS]):
        cohorts = [ps_to_cohort(self, ps) for ps in pss]

        all_rotas = np.vstack(self.extended_patterns)
        all_rotas = list(set(tuple(row) for row in all_rotas))
        all_skills = list(self.all_skills)

        def skills_of_cohort(cohort: Cohort) -> list[Skill]:
            return [skill for member in cohort
                    for skill in member.worker.available_skills]

        def rotas_of_cohort(cohort: Cohort) -> list[tuple]:
            return [tuple(member.chosen_rota_extended) for member in cohort]

        def rotas_of_worker(worker_index: int) -> list[tuple]:
            return [tuple(row) for row in self.extended_patterns[worker_index]]

        skill_distribution_in_pss = utils.count_frequency_in_containers(map(skills_of_cohort, cohorts),
                                                                        all_skills)
        skill_distribution_in_problem = utils.count_frequency_in_containers(
            map(lambda w: w.available_skills, self.workers),
            all_skills)
        rota_distribution_in_pss = utils.count_frequency_in_containers(map(rotas_of_cohort, cohorts),
                                                                       all_rotas)
        rota_distribution_in_problem = utils.count_frequency_in_containers(
            map(rotas_of_worker, range(len(self.workers))),
            all_rotas)

        def sort_by_delta(container: list):
            def key_func(item) -> float:
                return abs(item[2] - item[1])

            return sorted(container, key=key_func, reverse=True)

        skills_freq = list(zip(self.all_skills, skill_distribution_in_pss, skill_distribution_in_problem))
        skills_freq = sort_by_delta(skills_freq)

        # calculating how many times a rota was chosen vs available
        def rotas_in_solution(solution: FullSolution) -> list[tuple]:
            return [tuple(self.extended_patterns[var][val]) for var, val in enumerate(solution.values)]

        def rota_counts_in_solution(solution: FullSolution) -> np.ndarray:
            rotas_present = rotas_in_solution(solution)
            return np.array([len([1 for rota in rotas_present if rota == wanted_rota])
                             for wanted_rota in all_rotas])

        def winning_rate_for_solution(e_solution: EvaluatedFS) -> np.ndarray:
            counts = rota_counts_in_solution(e_solution.full_solution)
            total_per_rota = rota_distribution_in_problem * len(self.workers)
            return counts / total_per_rota

        def average_of_winning_rates(solutions: list[FullSolution]) -> np.ndarray:
            return np.average(np.array(list(map(winning_rate_for_solution, solutions))), axis=0)

        rota_winning_freqs = average_of_winning_rates(full_solutions)
        rota_freq = list(zip(all_rotas, rota_distribution_in_pss, rota_distribution_in_problem, rota_winning_freqs))
        rota_freq = sort_by_delta(rota_freq)

        def useful_properties(rota: tuple) -> list[float]:
            pattern = np.array(rota).reshape((-1, 7))
            covered_saturdays = np.sum(pattern[:, 5])
            covered_sundays = np.sum(pattern[:, 6])
            range_score = self.aggregate_range_scores(get_range_scores(pattern))
            return [covered_saturdays, covered_sundays, range_score]

        def as_percentage(num: float) -> str:
            return f"{num * 100:.2f}%"

        print(f"The skill distribution is")
        for skill, pss_freq, problem_freq in skills_freq:
            print(f"\t{skill}: \tps={as_percentage(pss_freq)}, \tprob={as_percentage(problem_freq)}")

        print(f"The rota distribution is")
        for rota, pss_freq, problem_freq, winning_freq in rota_freq:
            rota_str = "".join("-" if v == 0 else "W" for v in rota)
            adj_problem_freq = int(problem_freq * len(self.workers))
            adj_winning_freq = winning_freq * adj_problem_freq
            # print(f"\t{rota_str}: \tps={as_percentage(pss_freq)}, \tprob={as_percentage(problem_freq)}, \twins={as_percentage(winning_freq)}")
            properties = useful_properties(rota)
            print(
                f"\t{rota_str}: \tps={as_percentage(pss_freq)}, \t#workers={adj_problem_freq}, \tavg.wins={adj_winning_freq:.2f}, {properties =}")

        utils.simple_scatterplot("skill_pss_freq", "skill_problem_freq", skill_distribution_in_pss,
                                 skill_distribution_in_problem)
        utils.make_interactive_3d_plot(rota_distribution_in_pss,
                                       rota_distribution_in_problem,
                                       rota_winning_freqs,
                                       names=["rota_pss_freq", "rota_problem_freq", "rota_winning"])

        # filename = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\MartinBT\rota_popularity.csv"

    @classmethod
    def from_inverse_graph_colouring_problem(cls, original_problem: InverseGraphColouringProblem):

        if original_problem.amount_of_colours != 2:
            raise Exception("Only 2 colours are currently supported in the BT version of the IGC problem")

        workday = WorkDay.working_day(900, 1700)
        restday = WorkDay.not_working()

        rota_alpha = RotaPattern(7, [workday, workday, workday, restday, restday, restday, restday])
        rota_beta = RotaPattern(7, [restday, restday, restday, workday, workday, workday, restday])

        non_working_week = [restday] * 7

        def place_in_otherwise_empty_rota(weeks: int, rota_to_insert: RotaPattern, insertion_point: int) -> RotaPattern:
            return RotaPattern(7,
                               non_working_week * insertion_point +
                               rota_to_insert.days +
                               non_working_week * (weeks - insertion_point - 1))

        def options_for_nth_worker(n: int) -> list[RotaPattern]:
            return [place_in_otherwise_empty_rota(weeks=original_problem.clique_size,
                                                  rota_to_insert=rota_alpha,
                                                  insertion_point=n),
                    place_in_otherwise_empty_rota(weeks=original_problem.clique_size,
                                                  rota_to_insert=rota_beta,
                                                  insertion_point=n)]

        def make_template_worker(name: str, rota_options: list[RotaPattern]):
            return Worker(available_skills=set(),
                          available_rotas=rota_options,
                          worker_id=name,
                          name=name)

        def make_clique_for_skill(skill: str) -> list[Worker]:
            return [Worker(available_skills={skill},
                           available_rotas=options_for_nth_worker(n),
                           worker_id=f"W{n}_S{skill}",
                           name=f"W{n}_S{skill}")
                    for n in range(original_problem.clique_size)]

        workers = [worker for s in range(original_problem.amount_of_cliques)
                   for worker in make_clique_for_skill(f"SKILL_{s}")]

        return EfficientBTProblem(workers=workers,
                                  calendar_length=7 * original_problem.clique_size,
                                  weights=[1, 1, 1, 1, 1, 1, 0],
                                  rota_preference_weight=0)

    def get_short_code(self) -> str:
        return "BT"

    @classmethod
    def subset_from(cls,
                    original_problem,
                    which_workers_to_keep: list[int],
                    rota_lengths: Optional[list[int]],
                    calendar_length: int,
                    skills_conversion_dict: Optional[dict[str, str]]):
        assert (isinstance(original_problem, EfficientBTProblem))

        def worker_with_new_skills(original_worker) -> Worker:
            new_skills = {skills_conversion_dict[old_skill] for old_skill in original_worker.available_skills}
            return Worker(available_skills=new_skills,
                          available_rotas=original_worker.available_rotas,
                          name=original_worker.name,
                          worker_id=original_worker.worker_id)

        def cut_rota_pattern(old_rota_pattern: RotaPattern, rota_length: int) -> RotaPattern:
            new_days = []
            while len(new_days) <= rota_length:
                new_days.extend(old_rota_pattern.days)

            new_days = new_days[:rota_length]
            return RotaPattern(workweek_length=old_rota_pattern.workweek_length, days=new_days)

        def worker_with_new_rotas(original_worker: Worker, rota_length: int) -> Worker:
            new_rotas = [cut_rota_pattern(rota, rota_length)
                         for rota in original_worker.available_rotas]
            new_rotas = list(set(new_rotas))
            return Worker(available_skills=original_worker.available_skills,
                          available_rotas=new_rotas,
                          name=original_worker.name,
                          worker_id=original_worker.worker_id)

        workers = [original_problem.workers[index] for index in which_workers_to_keep]

        if skills_conversion_dict is not None:
            workers = list(map(worker_with_new_skills, workers))

        if rota_lengths is not None:
            workers = [worker_with_new_rotas(worker, rota_length)
                       for worker, rota_length in zip(workers, rota_lengths)]

        return cls(workers=workers,
                   calendar_length=calendar_length,
                   weights=original_problem.weights,
                   use_faulty_fitness_function=original_problem.use_faulty_fitness_function,
                   rota_preference_weight=original_problem.rota_preference_weight)

    @classmethod
    def random_subset_of(cls,
                         original_problem,
                         quantity_workers_to_keep: int,
                         skills_to_use: set[str],
                         max_rota_length: int,
                         calendar_length: int,
                         random_state: int = 120):
        amount_of_workers_in_total = len(original_problem.workers)

        random.seed(random_state)
        which_workers_to_keep = random.sample(population=range(amount_of_workers_in_total), k=quantity_workers_to_keep)
        skills_to_choose_from = sorted(list(skills_to_use))

        random.seed(random_state)
        skills_conversion_dict = {original_skill: random.choice(skills_to_choose_from)
                                  for original_skill in sorted(original_problem.all_skills)}

        random.seed(random_state)
        rota_lengths = [random.randrange(1, max_rota_length + 1) * 7 for _ in which_workers_to_keep]
        return cls.subset_from(original_problem=original_problem,
                               which_workers_to_keep=which_workers_to_keep,
                               skills_conversion_dict=skills_conversion_dict,
                               calendar_length=calendar_length,
                               rota_lengths=rota_lengths)

    def print_for_google_sheets(self):
        all_rotas = list(set(rota for worker in self.workers
                             for rota in worker.available_rotas))

        ordered_skills = list(self.all_skills)

        def get_rota_name(rota: RotaPattern) -> str:
            index = all_rotas.index(rota)
            return f"{index + 1}"

        def get_rota_representation(rota: RotaPattern) -> str:
            return "\t".join(f"{day}" for day in rota.days)

        def get_skillset_repr(skillset: set[str]) -> str:
            return "\t".join(("" if skill in skillset else skill) for skill in ordered_skills)

        def get_worker_representation(worker: Worker) -> str:
            skills_str = get_skillset_repr(worker.available_skills)
            rotas_str = "\t".join(get_rota_name(rota) for rota in worker.available_rotas)
            return "\t".join([worker.name, skills_str, rotas_str])

        # print the workers
        for worker in self.workers:
            print(get_worker_representation(worker))

        print("\n" * 4)

        # print the rotas
        for rota in all_rotas:
            print("\t".join([get_rota_name(rota), get_rota_representation(rota)]))

    def print_ps_for_google_sheets(self, ps: PS):
        def get_items_from_fixed_variable(index, value) -> dict:
            return {"worker": self.workers[index],
                    "chosen_rota_index": value,
                    "chosen_rota": self.extended_patterns[index][value]}

        items = [get_items_from_fixed_variable(index, value)
                 for index, value in enumerate(ps.values) if value != STAR]

        def get_rota_representation(rota: np.ndarray) -> str:
            return "\t".join("W" if working else "-" for working in rota)

        def item_to_string(item: dict):
            return "\t".join([item["worker"].name,
                              f"{utils.alphabet[item['chosen_rota_index']]}",
                              get_rota_representation(item["chosen_rota"])
                              ])

        for item in items:
            print(item_to_string(item))

    def breakdown_of_fitness_function(self, solution: FullSolution) -> dict:
        chosen_patterns = self.get_chosen_patterns_from_fs(solution)
        quantity_of_unliked_rotas = np.sum(solution.values != 0)

        result = {"quantity_unliked_rotas": quantity_of_unliked_rotas,
                  "by_skill": {skill: None for skill in self.all_skills},
                  "by_weekday": {weekday: None for weekday in utils.weekdays},
                  "by_skill_and_weekday": {
                      skill: {weekday: None for weekday in utils.weekdays}
                      for skill in self.all_skills
                  }
                  }

        for skill in self.all_skills:
            ranges = self.get_ranges_for_weekdays_for_skill(chosen_patterns, skill)
            result["by_skill"][skill] = self.aggregate_range_scores(ranges)
            for range_score, weekday in zip(ranges, utils.weekdays):
                result["by_skill_and_weekday"][skill][weekday] = range_score

        for weekday in utils.weekdays:
            result["by_weekday"][weekday] = sum(result["by_skill_and_weekday"][skill][weekday]
                                                for skill in self.all_skills)

        return result

    def get_minimums_and_maximums_dict_for_fs(self, fs: FullSolution) -> dict[str, dict[str, dict[str, int]]]:
        # the dict is [skill][weekday]["minimum" | "maximum"]
        # patterns_by_skill = {skill: [] for skill in self.all_skills}
        workers_and_rotas = [(self.workers[index], self.extended_patterns[index][chosen])
                             for index, chosen in enumerate(fs.values)]

        patterns_by_skill = {skill: [rota for (worker, rota)
                                     in workers_and_rotas
                                     if skill in worker.available_skills]
                             for skill in self.all_skills}

        def get_maxmin_dict_for_patterns(patterns: list[np.ndarray]):
            summed_pattern = np.sum(patterns, axis=0)
            coiled_pattern = summed_pattern.reshape((-1, 7))
            maxs = np.max(coiled_pattern, axis=0)
            mins = np.min(coiled_pattern, axis=0)
            return {weekday: {"maximum": maximum, "minimum": minimum}
                    for weekday, maximum, minimum in zip(utils.weekdays, maxs, mins)}

        return {skill: get_maxmin_dict_for_patterns(patterns_by_skill[skill])
                for skill in patterns_by_skill}