import json
import os
from typing import TypeAlias

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace

Clause: TypeAlias = np.ndarray


class SATProblem(BenchmarkProblem):
    amount_of_variables: int
    amount_of_clauses: int
    solvable: bool

    clauses: list[Clause]

    def __init__(self,
                 amount_of_variables: int,
                 amount_of_clauses: int,
                 solvable: bool,
                 clauses: list[Clause]):

        self.amount_of_variables = amount_of_variables
        self.amount_of_clauses = amount_of_clauses
        self.solvable = solvable
        self.clauses = clauses

        assert (amount_of_clauses == len(self.clauses))
        search_space = SearchSpace([2 for _ in range(self.amount_of_variables)])
        super().__init__(search_space)

    def long_repr(self) -> str:
        def repr_var(var_number):
            if var_number < 0:
                return f"NOT(var{utils.alphabet[(-var_number) - 1]})"
            else:
                return f"var{utils.alphabet[var_number]}"

        def repr_clause(clause: Clause):
            numbers = SATProblem.clause_to_numbers(clause)
            return " OR ".join(repr_var(var_number) for var_number in numbers)

        result = repr(self) + "clauses are :"
        result += "\n".join(repr_clause(clause) for clause in self.clauses)

        return result

    def repr_ps(self, ps: PS) -> str:
        return ", ".join(f"{utils.alphabet[var]}={val}" for var, val in enumerate(ps.values) if val != STAR)

    @staticmethod
    def numbers_to_clause(numbers: list[int], amount_of_variables: int) -> Clause:
        clause = np.zeros(amount_of_variables, dtype=int)
        for number in numbers:
            clause[abs(number) - 1] = 1 if number > 0 else -1
        return clause

    @staticmethod
    def clause_to_numbers(clause: Clause) -> list[int]:
        def get_number(index, value):
            if value == 1:
                return index + 1
            elif value == -1:
                return -(index + 1)

        return [get_number(index, value)
                for index, value in enumerate(clause)
                if value != 0]

    @classmethod
    def from_cnf_file(cls, cnf_file_location: str):
        # get them from https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html
        filename = cnf_file_location.split("\\")[-1]
        solvable = filename[:2] == "uf"

        amount_of_variables = None
        amount_of_clauses = None
        clause_reading_mode = False
        current_clause_buffer: list[int] = []
        clauses = []

        def consume_first_available_clause_from_buffer(current_clause_buffer) -> (list[int], Clause):
            position_of_zero = current_clause_buffer.index(0)
            if position_of_zero is not None:
                numbers_to_convert = current_clause_buffer[:position_of_zero]
                new_clause = SATProblem.numbers_to_clause(numbers_to_convert, amount_of_variables)
                new_buffer = current_clause_buffer[(position_of_zero + 1):]
                return new_buffer, new_clause

        def consume_from_buffer(current_clause_buffer) -> list[int]:
            while 0 in current_clause_buffer:
                current_clause_buffer, new_clause = consume_first_available_clause_from_buffer(current_clause_buffer)
                clauses.append(new_clause)
            return current_clause_buffer

        with open(cnf_file_location, "r") as file:

            for line in file.readlines():
                if len(line) == 0:
                    # print("Found an empty line, skipping")
                    continue

                if line[:1] == "c":
                    # print("Found a comment line, skipping")
                    continue

                if clause_reading_mode:
                    try:
                        numbers_in_line = [int(value_str) for value_str in line.split()]
                    except ValueError:
                        print(f"There was an error when reading the line {line} (in clause mode)")

                    current_clause_buffer.extend(numbers_in_line)
                    current_clause_buffer = consume_from_buffer(current_clause_buffer)
                    if len(clauses) == amount_of_clauses:
                        # print("Read all of the specified clauses, file interpret will terminate")
                        break

                elif line[:1] == "p":
                    # print("Found the problem line")
                    p_char, problem_kind, var_str, clause_str = line.split()
                    if problem_kind != "cnf":
                        print(f"Error! The problem kind is not cnf, but {problem_kind}! Terminating")
                        raise ValueError

                    try:
                        amount_of_variables = int(var_str)
                        amount_of_clauses = int(clause_str)
                        clause_reading_mode = True
                    except ValueError:
                        print(f"There was an error when reading the line {line} in clause reading mode")
                else:
                    print(f"The line {line} was not recognised, terminating")

            return cls(amount_of_variables=amount_of_variables,
                       amount_of_clauses=amount_of_clauses,
                       solvable=solvable,
                       clauses=clauses)

    @classmethod
    def from_json_file(cls, json_file_location: str):
        with open(json_file_location, "r") as file:
            data = json.load(file)
            amount_of_variables = data["amount_of_variables"]
            return cls(solvable=data["solvable"],
                       amount_of_clauses=data["amount_of_clauses"],
                       amount_of_variables=amount_of_variables,
                       clauses=[SATProblem.numbers_to_clause(numbers, amount_of_variables)
                                for numbers in data["clauses"]])

    def to_json_file(self, file_location: str):
        result = dict()
        result["amount_of_variables"] = self.amount_of_variables
        result["amount_of_clauses"] = self.amount_of_clauses
        result["solvable"] = self.solvable
        result["clauses"] = [SATProblem.clause_to_numbers(clause) for clause in self.clauses]

        with open(file_location, "w+") as output_file:
            json.dump(result, output_file, indent=4)

    def fitness_function(self, fs: FullSolution) -> float:

        reworked_values = fs.values * 2 - 1  # so that they're +-1

        def clause_is_satisfied(clause: Clause):
            return any(reworked_values * clause == 1)

        return float(sum(clause_is_satisfied(clause) for clause in self.clauses))

    def __repr__(self):
        return f"SATProblem(#vars = {self.amount_of_variables}, #clauses = {self.amount_of_clauses})"

    def get_global_optima_fitness(self) -> float:
        if self.solvable:
            return self.amount_of_clauses
        else:
            return np.nan


# convert_problem_files_from_cnf()


class SATExplainer:
    problem: SATProblem

    univariate_counts: np.ndarray
    bivariate_counts: np.ndarray

    def __init__(self, problem: SATProblem):
        self.problem = problem
        self.univariate_counts = self.get_univariate_counts(problem)
        self.bivariate_counts = self.get_bivariate_counts(problem)

    @classmethod
    def get_clause_matrix(cls, problem: SATProblem) -> np.ndarray:
        return np.abs(np.array([clause for clause in problem.clauses], dtype=int))

    @classmethod
    def get_univariate_counts(cls, problem: SATProblem) -> np.ndarray:
        return np.sum(cls.get_clause_matrix(problem), axis=0)

    def get_bivariate_counts(self, problem: SATProblem) -> np.ndarray:
        return sum([np.outer(clause, clause) for clause in self.get_clause_matrix(problem)])
