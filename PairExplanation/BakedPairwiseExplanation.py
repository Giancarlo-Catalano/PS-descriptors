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

    def __init__(self,
                 main_solution: FullSolution,
                 background_solution: FullSolution,
                 difference_pattern: PS,
                 explanation_text: str,
                 descriptor_tuple: list[(str, float, float)]):
        self.main_solution = main_solution
        self.background_solution = background_solution
        self.difference_pattern = difference_pattern
        self.explanation_text = explanation_text
        self.descriptor_dict = descriptor_tuple

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

