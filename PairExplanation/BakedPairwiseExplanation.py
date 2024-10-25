from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import WilcoxonTest, WilcoxonNearOptima, \
    get_hypothesis_string
from PairExplanation.BTProblemPrettyPrinter import BTProblemPrettyPrinter


class BakedPairwiseExplanation:
    main_solution: FullSolution
    background_solution: FullSolution
    difference_pattern: PS
    explanation_text: str
    descriptor_dict: list[(str, float, float)]
    label: str

    def __init__(self,
                 main_solution: FullSolution,
                 background_solution: FullSolution,
                 difference_pattern: PS,
                 explanation_text: str,
                 descriptor_tuples: list[(str, float, float)],
                 label: str = "no label"):
        self.main_solution = main_solution
        self.background_solution = background_solution
        self.difference_pattern = difference_pattern
        self.explanation_text = explanation_text
        self.descriptor_dict = descriptor_tuples
        self.label = label

    def print_using_pretty_printer(self,
                                   pretty_printer: BTProblemPrettyPrinter,
                                   hypothesis_tester: WilcoxonTest,
                                   near_optima_hypothesis_tester: WilcoxonNearOptima,
                                   show_solutions: bool = False):

        if show_solutions:
            print("main solution = ")
            print(pretty_printer.repr_full_solution(self.main_solution))
            print(pretty_printer.repr_extra_information_for_full_solution(self.main_solution))

            print("background solution = ")
            print(pretty_printer.repr_full_solution(self.background_solution))
            print(pretty_printer.repr_extra_information_for_full_solution(self.background_solution))

        print("The difference between the solutions is ")
        print(pretty_printer.repr_difference_between_solutions(self.main_solution,
                                                               self.background_solution))

        print("Partial solution = ")
        print(pretty_printer.repr_partial_solution(self.difference_pattern))
        print(pretty_printer.repr_extra_information_for_partial_solution(self.difference_pattern,
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
        print(problem.repr_ps(self.difference_pattern))
        print(get_hypothesis_string(self.difference_pattern,
                                    hypothesis_tester,
                                    near_optima_hypothesis_tester))

        print("Explanation string")
        print(self.explanation_text)


    def get_difference_in_rotas_table(self, pretty_printer: BTProblemPrettyPrinter):
        different_variable_indexes = [index for index, is_different
                                      in enumerate(self.main_solution.values != self.background_solution)
                                      if is_different]

        for var_index in different_variable_indexes:
            rota_choice_in_main = self.main_solution.values[var_index]
            rota_in_background = self.background_solution.values[var_index]

            worker_name = pretty_printer.get_worker_name(var_index)
            rota_index_main_label = pretty_printer.get_value_as_rota_index(var_index, rota_choice_in_main)
            rota_index_background_label = pretty_printer.get_value_as_rota_index(var_index, rota_in_background)

            print("\t".join([worker_name,
                             f"{rota_choice_in_main} = {rota_index_main_label}",
                             f"{rota_in_background} = {rota_index_background_label}"
                             ]))




    def to_json(self) -> dict:
        return {"main_solution": self.main_solution.to_json(),
                "background_solution": self.background_solution.to_json(),
                "difference_pattern": self.difference_pattern.to_json(),
                "descriptor_tuples": self.descriptor_dict,
                "explanation_text": self.explanation_text,
                "label": self.label}

    @classmethod
    def from_json(cls, json_dict: dict):
        main_solution = FullSolution.from_json(json_dict["main_solution"])
        background_solution = FullSolution.from_json(json_dict["background_solution"])
        difference_pattern = PS.from_json(json_dict["difference_pattern"])
        return cls(main_solution=main_solution,
                   background_solution = background_solution,
                   difference_pattern = difference_pattern,
                   descriptor_tuples= json_dict["descriptor_tuples"],
                   explanation_text=json_dict["explanation_text"],
                   label = json_dict["label"])

