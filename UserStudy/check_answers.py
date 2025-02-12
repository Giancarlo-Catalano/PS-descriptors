import itertools

import utils
from BenchmarkProblems.SimplifiedBTProblem.SimplifiedBTProblem import SimplifiedBTProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from UserStudy.problem_and_explanations_script import get_problem_path, get_pRef_path


def check_answers():
    instance_cb_path = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\UserStudy\Instances\Constructed_B"
    problem_b_path = get_problem_path(instance_cb_path)
    pRef_path = get_pRef_path(instance_cb_path)

    problem = SimplifiedBTProblem.from_json(problem_b_path)
    pRef = PRef.load(pRef_path)

    def with_single_modification(solution: FullSolution, name_and_rota) -> FullSolution:
        name, rota = name_and_rota
        worker_index = problem.worker_names.index(name)
        rota_index = utils.alphabet.index(rota)

        return solution.with_different_value(worker_index, rota_index)



    def with_modifications(solution: FullSolution, modifications) -> EvaluatedFS:
        new_solution = solution.copy()
        for item in modifications:
            new_solution = with_single_modification(new_solution, item)
        return EvaluatedFS(new_solution, fitness=problem.fitness_function(new_solution))

    best_solution = "CCDCDDCCDCDCCCDDDA"
    #best_solution = "AAAAAAAAAAAAAAAAAA"
    best_solution = FullSolution(utils.alphabet.index(letter) for letter in best_solution)
    best_solution = EvaluatedFS(best_solution, problem.fitness_function(best_solution))
    print(f"The best solution has fitness {best_solution.fitness}, and its {problem.repr_ps(PS.from_FS(best_solution))}")



    task_1_modifications = [
        [("Brandon", "D"), ("Kevin", "C")],
        [("Brandon", "D")],
        [("Phoebe", "C")],
        [("Phoebe", "C"), ("Brandon", "D")]
    ]

    task_2_modifications = [[(x, "C"), (y, "C")]
                            for x, y in itertools.combinations(["Clara", "Fiona", "Rose", "Ivy", "Eleanor"], r=2)]



    def show_modification_ordered(mods):
        modified_solutions = [(modifs, with_modifications(best_solution, modifs))
                              for modifs in mods]
        modified_solutions.sort(key=lambda x: x[1].fitness, reverse=True)

        for modif, solution in modified_solutions:
            print(f"With modification {modif}, the fitness is {solution.fitness}")


    print("For task 1")
    show_modification_ordered(task_1_modifications)


    print("For task 2")
    show_modification_ordered(task_2_modifications)


check_answers()