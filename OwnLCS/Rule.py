from typing import TypeAlias

from Core.FullSolution import FullSolution
from Core.PS import PS, contains

OutputClass: TypeAlias = int


class Rule:
    ps: PS
    predicted_class: OutputClass

    numerosity: int
    amount_of_matches: int
    amount_of_correct: int

    cached_linkage: float
    cached_simplicity: float


    def __init__(self,
                 ps: PS,
                 predicted_class: OutputClass,
                 numerosity: int = 1,
                 amount_of_matches: int = 0,
                 amount_of_correct: int = 0):
        self.ps = ps
        self.predicted_class = predicted_class

        self.numerosity = numerosity
        self.amount_of_matches = amount_of_matches
        self.amount_of_correct = amount_of_correct

        self.cached_linkage = 0
        self.cached_simplicity = 0

    def __repr__(self) -> str:
        return (f"{self.ps} -> {self.predicted_class} "
                f"(numerosity = {self.numerosity}, "
                f"accuracy = {int(self.get_accuracy()*100)}%, "
                f"#C = {self.amount_of_correct}, "
                f"#M = {self.amount_of_matches})")

    def matches(self, full_solution: FullSolution) -> bool:
        return contains(full_solution, self.ps)

    def get_accuracy(self) -> float:
        if self.amount_of_matches < 1:
            return 0
        else:
            return self.amount_of_correct / self.amount_of_matches
