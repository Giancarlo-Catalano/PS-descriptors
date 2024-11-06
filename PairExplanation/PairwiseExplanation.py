import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima, \
    get_hypothesis_string
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter


class PairwiseExplanation:
    main_solution: FullSolution
    background_solution: FullSolution
    partial_solution: PS
    explanation_text: str
    descriptor_tuples: list[(str, float, float)]
    label: str

    def __init__(self,
                 main_solution: FullSolution,
                 background_solution: FullSolution,
                 partial_solution: PS,
                 explanation_text: str,
                 descriptor_tuples: list[(str, float, float)],
                 label: str = "no label"):
        self.main_solution = main_solution
        self.background_solution = background_solution
        self.partial_solution = partial_solution
        self.explanation_text = explanation_text
        self.descriptor_tuples = descriptor_tuples
        self.label = label

    def print_using_pretty_printer(self,
                                   pretty_printer: BTProblemPrettyPrinter,
                                   hypothesis_tester: WilcoxonTest,
                                   near_optima_hypothesis_tester: WilcoxonNearOptima,
                                   show_solutions: bool = False):

        if show_solutions:
            print("main solution = ")
            print(pretty_printer.repr_full_solution(self.main_solution))
            print("\n")
            print(pretty_printer.repr_extra_information_for_full_solution(self.main_solution))

            print("background solution = ")
            print(pretty_printer.repr_full_solution(self.background_solution))
            print("\n")
            print(pretty_printer.repr_extra_information_for_full_solution(self.background_solution))

        print("The difference between the solutions is ")
        print(pretty_printer.repr_difference_between_solutions(self.main_solution,
                                                               self.background_solution))

        print("Partial solution = ")
        print(pretty_printer.repr_partial_solution(self.partial_solution))
        print("\n")
        print(pretty_printer.repr_extra_information_for_partial_solution(self.partial_solution,
                                                                         hypothesis_tester,
                                                                         near_optima_hypothesis_tester))

        print("Explanation string")
        print(self.explanation_text)

        main_fitness = pretty_printer.problem.fitness_function(self.main_solution)
        background_fitness = pretty_printer.problem.fitness_function(self.background_solution)
        print(f"The fitnesses are main = {main_fitness}, background = {background_fitness}")

    def print_normally(self,
                       problem: BenchmarkProblem,
                       hypothesis_tester: WilcoxonTest,
                       near_optima_hypothesis_tester: WilcoxonNearOptima,
                       show_solutions: bool = False):

        main_fitness, background_fitness = [problem.fitness_function(s)
                                            for s in [self.main_solution, self.background_solution]]
        if show_solutions:
            print("main solution = ")
            print(problem.repr_full_solution(self.main_solution))
            print(f"It has fitness {main_fitness}")

            print("main solution = ")
            print(problem.repr_full_solution(self.background_solution))
            print(f"It has fitness {background_fitness}")

        print("Partial solution = ")
        print(problem.repr_ps(self.partial_solution))
        print(get_hypothesis_string(self.partial_solution,
                                    hypothesis_tester,
                                    near_optima_hypothesis_tester))

        print("Explanation string")
        print(self.explanation_text)


    def get_comparison_of_solution_on_variables(self, different_variables: list[int], pretty_printer: BTProblemPrettyPrinter) -> str:
        def get_row_for_index(var_index: int) -> str:
            rota_choice_in_main = self.main_solution.values[var_index]
            rota_in_background = self.background_solution.values[var_index]

            worker_name = pretty_printer.get_worker_name(var_index)
            rota_index_main_label = pretty_printer.get_value_as_rota_index(var_index, rota_choice_in_main)
            rota_index_background_label = pretty_printer.get_value_as_rota_index(var_index, rota_in_background)

            return "\t".join([worker_name,
                              f"{pretty_printer.repr_rota_choice(rota_choice_in_main)} = {rota_index_main_label}",
                              f"{pretty_printer.repr_rota_choice(rota_in_background)} = {rota_index_background_label}"
                              ])

        return "\n".join(map(get_row_for_index, different_variables))


    def get_difference_in_rotas_table(self, pretty_printer: BTProblemPrettyPrinter) -> str:
        different_variable_indexes = [index for index, is_different
                                      in enumerate(self.main_solution.values != self.background_solution.values)
                                      if is_different]

        return self.get_comparison_of_solution_on_variables(different_variable_indexes, pretty_printer)

    def get_ps_table(self, pretty_printer: BTProblemPrettyPrinter) -> str:
        different_variable_indexes = self.partial_solution.get_fixed_variable_positions()

        return self.get_comparison_of_solution_on_variables(different_variable_indexes, pretty_printer)

    def get_hypothesis_test_results(self, hypothesis_tester: WilcoxonTest) -> str:
        p_value_higher, p_value_lower = hypothesis_tester.get_p_values_of_ps(self.partial_solution)

        lower_significance = utils.get_p_value_significance(p_value_lower)
        higher_significance = utils.get_p_value_significance(p_value_higher)


        match (lower_significance, higher_significance):
            case ("INSIGNIFICANT", "INSIGNIFICANT"):
                return "The pattern was not found to be significantly positive or negative"
            case ("INSIGNIFICANT", _):
                return f"The pattern was found to be beneficial ({higher_significance}), with p_value {p_value_higher}"
            case (_, "INSIGNIFICANT"):
                return f"The pattern was found to be negative ({lower_significance}), with p_value {p_value_lower}"
            case _:
                return f"Somehow, the pattern is both positive and negative ({p_value_lower = }, {p_value_higher})"

    def to_json(self) -> dict:
        return {"main_solution": self.main_solution.to_json(),
                "background_solution": self.background_solution.to_json(),
                "difference_pattern": self.partial_solution.to_json(),
                "descriptor_tuples": self.descriptor_tuples,
                "explanation_text": self.explanation_text,
                "label": self.label}

    @classmethod
    def from_json(cls, json_dict: dict):
        main_solution = FullSolution.from_json(json_dict["main_solution"])
        background_solution = FullSolution.from_json(json_dict["background_solution"])
        difference_pattern = PS.from_json(json_dict["difference_pattern"])
        return cls(main_solution=main_solution,
                   background_solution=background_solution,
                   partial_solution=difference_pattern,
                   descriptor_tuples=json_dict["descriptor_tuples"],
                   explanation_text=json_dict["explanation_text"],
                   label=json_dict["label"])

    def get_changes_in_calendar(self, pretty_printer: BTProblemPrettyPrinter) -> str:
        main_solution_calendar_counts = pretty_printer.get_calendar_counts_for_ps(PS.from_FS(self.main_solution))
        back_solution_calendar_counts = pretty_printer.get_calendar_counts_for_ps(PS.from_FS(self.background_solution))

        def aggregate_difference_tuples(diffs: list[(str, [int], int, int, int)]) -> list[(str, [int], int, int, int)]:
            """Not my best work..."""
            skill_before_after_weekday_dict = {(skill, before, after, weekday): []
                                               for skill, weeks, weekday, before, after in diffs}

            for skill, weeks, weekday, before, after in diffs:
                skill_before_after_weekday_dict[(skill, before, after, weekday)].extend(weeks)

            return [(key[0], weeks, key[3], key[1], key[2]) for key, weeks in skill_before_after_weekday_dict.items()]



        def get_differences_for_skill(skill: str) -> list[(str, [int], int, int, int)]:
            main_calendar = main_solution_calendar_counts[skill]
            back_calendar = back_solution_calendar_counts[skill]
            calendar_differences: np.ndarray = main_calendar != back_calendar
            different_day_indices = [index for index, is_different in enumerate(calendar_differences) if is_different]

            diffs = [(skill, [day // 7], day % 7, back_calendar[day], main_calendar[day]) for day in different_day_indices]
            return aggregate_difference_tuples(diffs)

        def repr_difference(diff_tuple: (str, [int], int, int, int)) -> str:
            # note that before, after = background, main
            skill, weeks, weekday, before, after = diff_tuple
            weeks_str = "Week"+("s" if len(weeks)>1 else "")+ " "+ ",".join(f"{w+1}" for w in weeks)
            weekday_str = utils.weekdays[weekday]
            return "\t".join([pretty_printer.repr_skill(skill),
                              weeks_str,
                             weekday_str,
                              f"{after}",
                              f"{before}"])

        all_differences = [diff_tuple
                           for skill in pretty_printer.all_skills_list
                           for diff_tuple in get_differences_for_skill(skill)]

        return "\n".join(map(repr_difference, all_differences))

    def get_changes_in_range(self, pretty_printer: BTProblemPrettyPrinter) -> str:
        main_maxmins = pretty_printer.problem.get_minimums_and_maximums_dict_for_fs(self.main_solution)
        back_maxmins = pretty_printer.problem.get_minimums_and_maximums_dict_for_fs(self.background_solution)


        differences = []

        for skill in pretty_printer.all_skills_list:
            for weekday in utils.weekdays:
                min_in_main = main_maxmins[skill][weekday]["minimum"]
                min_in_back = back_maxmins[skill][weekday]["minimum"]
                max_in_main = main_maxmins[skill][weekday]["maximum"]
                max_in_back = back_maxmins[skill][weekday]["maximum"]
                if (min_in_main, max_in_main) != (min_in_back, max_in_back):
                    if (min_in_main==max_in_main) and (min_in_back == max_in_back):
                        continue
                    differences.append((skill, weekday, min_in_main, max_in_main, min_in_back, max_in_back))

        def repr_difference(big_tuple) -> str:
            (skill, weekday, min_in_main, max_in_main, min_in_back, max_in_back) = big_tuple
            return "\t".join([pretty_printer.repr_skill(skill),
                              weekday,
                              f"{min_in_main}~{max_in_main}",
                              f"{min_in_back}~{max_in_back}"])

        return "\n".join(map(repr_difference, differences))




