from Core.FullSolution import FullSolution
from Core.PS import PS
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




    def print_using_pretty_printer(self, pretty_printer: BTProblemPrettyPrinter):

        print("main solution = ")
        print(pretty_printer.repr_full_solution(self.main_solution))
        print(pretty_printer.repr_extra_information_for_full_solution(self.main_solution))

        print("background solution = ")
        print(pretty_printer.repr_full_solution(self.background_solution))
        print(pretty_printer.repr_extra_information_for_full_solution(self.background_solution))

        print("Partial solution = ")
        print(pretty_printer.repr_partial_solution(self.difference_pattern))
        print(pretty_printer.repr_extra_information_for_partial_solution(self.difference_pattern))


        print("Explanation string")
        print(self.explanation_text)
