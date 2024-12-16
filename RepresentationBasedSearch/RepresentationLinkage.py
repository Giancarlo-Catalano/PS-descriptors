import itertools
from typing import Optional

import numpy as np
from tqdm import tqdm

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from FindClosestSolutionBasedOnRepresentation import find_solution_with_closest_representation
from ProblemRepresentation import Representation, ProblemRepresentation


class RepresentationLinkage(TraditionalPerturbationLinkage):
    # this exists because TraditionalPerturbationLinkage is the best you can get, but you can't get it for representations...

    current_solution_representation: Optional[Representation]
    current_solution: Optional[EvaluatedFS]
    background_solution_dictionary: Optional[dict[tuple, (EvaluatedFS, Representation)]]

    problem_representation: ProblemRepresentation


    def __init__(self,
                 optimisation_problem: BenchmarkProblem,
                 problem_representation: ProblemRepresentation):
        self.problem_representation = problem_representation
        self.current_solution_representation = None
        self.background_solution_dictionary = None
        super().__init__(optimisation_problem=optimisation_problem)

    def make_evaluated_fs(self, fs: FullSolution) -> EvaluatedFS:
        return EvaluatedFS(fs, fitness=self.optimisation_problem.fitness_function(fs))


    def set_solution(self, new_solution: FullSolution):
        evaluated_fs = self.make_evaluated_fs(new_solution)
        self.current_solution = evaluated_fs
        fs_representation = self.problem_representation.get_representation(new_solution)
        self.current_solution_representation = fs_representation

        self.background_solution_dictionary = self.find_all_background_solutions(main_solution = evaluated_fs,
                                                                                 main_representation = fs_representation)

        self.linkage_table = self.generate_linkage_table_from_background_solutions(main_representation = fs_representation,
                                                                                   background_representations= self.background_solution_dictionary)


    def get_representation_with_modification(self, representation: Representation, modification: tuple) -> Representation:
        result_values = representation.values.copy()
        for (var, val) in modification:
            result_values[var] = val
        return FullSolution(result_values)


    def get_ps_for_modification(self, modification: tuple) -> PS:
        result_values = PS.empty(search_space=self.problem_representation.representation_search_space).values
        for (var, val) in modification:
            result_values[var] = val
        return PS(result_values)

    def find_all_background_solutions(self, main_solution: EvaluatedFS, main_representation: Representation) -> dict[tuple, (EvaluatedFS, Representation)]:

        def get_entry_for_modification(modification: tuple) -> (EvaluatedFS, Representation):
            target_representation = self.get_representation_with_modification(main_representation, modification)
            parts_that_have_to_match = self.get_ps_for_modification(modification)
            solution, representation = find_solution_with_closest_representation(target_representation=target_representation,
                                                                                 parts_that_have_to_match=parts_that_have_to_match,
                                                                                 original_problem=self.optimisation_problem,
                                                                                 representation=self.problem_representation,
                                                                                 population_size=50,
                                                                                 search_budget=1000)
            return self.make_evaluated_fs(solution), representation

        rss = self.problem_representation.representation_search_space
        changes_dict = dict()

        ## univariate_changes
        for var, cardinality in enumerate(rss.cardinalities):
            for new_value in range(cardinality):
                modification = ((var, new_value),)
                found_solution, found_repr = get_entry_for_modification(modification)
                changes_dict[modification] = (found_solution, found_repr)

        # bivariate_changes
        for var_a, var_b in tqdm(itertools.combinations(range(rss.amount_of_parameters), r=2)):
            for val_a in range(rss.cardinalities[var_a]):
                for val_b in range(rss.cardinalities[var_b]):
                    modification = ((var_a, val_a), (var_b, val_b))
                    found_solution, found_repr = get_entry_for_modification(modification)
                    changes_dict[modification] = (found_solution, found_repr)

        changes_dict[()] = (main_solution, main_representation)

        return changes_dict

    def generate_linkage_table_from_background_solutions(self,
                                                         main_representation: Representation,
                                                         background_representations: dict) -> np.ndarray:
        own_fitness = background_representations[()][0].fitness

        def get_linkage_between_vals(var_a, val_a, var_b, val_b) -> float:
            both = background_representations[(var_a, val_a), (var_b, val_b)]
            just_a = background_representations[(var_a, val_a)]
            just_b = background_representations[(var_b, val_b)]

            return abs(own_fitness + both - just_a - just_b)

        rss = self.problem_representation.representation_search_space
        def get_linkage_between_vars(var_a, var_b) -> float:
            own_val_a = main_representation.values[var_a]
            own_val_b = main_representation.values[var_b]

            all_linkages = [get_linkage_between_vals(var_a, val_a, var_b, val_b)
                            for val_a in range(rss.cardinalities[var_a])
                            for val_b in range(rss.cardinalities[var_b])
                            if val_a != own_val_a
                            if val_b != own_val_b]
            if len(all_linkages) == 0:
                return 0

            return np.average(all_linkages)

        def get_importance_of_var(var_a: int) -> float:
            own_val_a = main_representation.values[var_a]
            other_fitnesses = [background_representations[(var_a, val_a)]
                               for val_a in range(rss.cardinalities[var_a])
                               if val_a != own_val_a]
            if len(other_fitnesses) == 0:
                return 0

            return np.average([abs(other_fit - own_fitness) for other_fit in other_fitnesses])

        n = rss.amount_of_parameters
        linkage_table = np.zeros(shape=(n, n))
        for var_a, var_b in itertools.combinations(range(n), r=2):
            linkage_table[var_a, var_b] = get_linkage_between_vars(var_a, var_b)

        linkage_table += linkage_table.T

        for var_a in range(n):
            linkage_table[var_a, var_a] = get_importance_of_var(var_a)

        return linkage_table

