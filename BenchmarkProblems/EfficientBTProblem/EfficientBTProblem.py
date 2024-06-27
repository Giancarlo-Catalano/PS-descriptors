import numpy as np
import pandas as pd

import utils
from BenchmarkProblems import InverseGraphColouringProblem
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_range_scores, WorkDay
from BenchmarkProblems.BT.Worker import Worker, Skill
from BenchmarkProblems.EfficientBTProblem.Cohort import ExtendedPattern, convert_worker_to_just_options, \
    FullPatternOptions, WeekRanges, Cohort, ps_to_cohort, get_skill_variation, get_skill_coverage, \
    get_hamming_distances, get_ranges_in_weekdays, get_mins_and_maxs_for_weekdays, get_coverage, \
    get_amount_of_covered_weekends
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from FirstPaper.EvaluatedFS import EvaluatedFS
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS


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
                                                  axis=0)  # not np.sum because it doesn't support generators
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

    def ps_to_properties(self, ps: PS) -> dict:
        cohort = ps_to_cohort(self, ps)

        choice_amounts = [member.get_amount_of_choices() for member in cohort]
        weekly_working_days = [member.get_mean_weekly_working_days() for member in cohort]
        rota_differences = get_hamming_distances(cohort)

        quantity_of_fav_rotas = len([worker for worker in cohort if worker.chosen_rota_index == 0])

        mean_RD, mean_error_RD = np.average(rota_differences)
        covered_saturdays, covered_sundays = get_amount_of_covered_weekends(cohort)

        skill_diversity = get_skill_variation(cohort)
        skill_coverage = get_skill_coverage(cohort)

        return {  # "mean_RCQ" : mean_RCQ,
            # "mean_error_RCQ" : mean_error_RCQ,
            # "mean_WWD" : mean_WWD,
            # "mean_error_WWD" : mean_error_WWD,
            "mean_RD": mean_RD,
            # "mean_error_RD" : mean_error_RD,
            "covered_sats": covered_saturdays,
            "covered_suns": covered_sundays,
            # "mean_error_WSP" : mean_error_WSP,
            # "mean_SQ": mean_SQ,
            # "mean_error_SQ": mean_error_SQ,
            "skill_diversity": skill_diversity,
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
        rank_str = f"(pol. = {int(rank * 100)}%)"  # "~ {int(property_rank_range[1]*100)}%)"

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
        else:
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

        # filename = r"C:\Users\gac8\PycharmProjects\PS-PDF\ExplanatoryCachedData\BT\MartinBT\rota_popularity.csv"

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

    def repr_fs(self, full_solution: FullSolution) -> str:
        old_repr = super().repr_fs(full_solution)
        chosen_patterns = self.get_chosen_patterns_from_fs(full_solution)
        quantity_of_unliked_rotas = np.sum(full_solution.values != 0)

        def ranges_for_skill(skill) -> list[(int, int)]:
            indexes = self.workers_by_skills[skill]
            summed_patterns: ExtendedPattern = np.sum([chosen_patterns[index] for index in indexes],
                                                      axis=0)  # not np.sum because it doesn't support generators
            summed_patterns = summed_patterns.reshape((-1, 7))
            mins = np.min(summed_patterns, axis=0)
            maxs = np.max(summed_patterns, axis=0)

            return list(zip(mins, maxs))

        final_str = old_repr + "\n" + f"{quantity_of_unliked_rotas = }\n"
        for skill in sorted(self.all_skills):
            final_str += f"{skill}:{ranges_for_skill(skill)}\n"

        return final_str

    def get_present_skills_in_pss(self, pss: list[PS]) -> pd.DataFrame:
        all_skills = [f"SKILL_{i}" for i in range(len(self.all_skills))]  # so that they are in order

        def get_which_skills_are_present_in_ps(ps: PS) -> np.ndarray:
            cohort = ps_to_cohort(self, ps)
            skills_present = set(skill for component in cohort for skill in component.worker.available_skills)
            return np.array([skill in skills_present for skill in all_skills])

        main_matrix = np.array([get_which_skills_are_present_in_ps(ps) for ps in pss])
        return pd.DataFrame(main_matrix, columns=all_skills)
