import json

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from PSMiners.Mining import write_pss_to_file

top_solution = FullSolution([0, 1, 0, 0, 0, 1, 3, 2, 0, 3, 0, 1, 1, 0, 0, 0, 1, 3, 0, 0, 0, 1, 0, 0, 2, 1, 3, 0, 0, 1, 0, 2, 0, 2, 1, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0, 3, 1, 2, 0, 3, 1, 0, 0, 0, 3, 0, 3, 3, 1, 1, 1, 0, 0, 2, 1, 1, 1, 3, 0, 3, 1, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 2, 1, 0, 1, 2, 0, 0, 0, 0, 3, 2, 1, 0, 1, 1, 0, 0, 0, 1, 3, 3, 0, 0, 2, 0, 1, 0, 1, 0, 1, 1, 2, 3])

problem = EfficientBTProblem.from_default_files()
def reverse_engineer_ps(names_of_workers: list[str]) -> PS:
    result = PS.empty(problem.search_space)
    def register_for_worker(name: str) -> (int, int):
        found_workers = [(index, worker) for (index, worker) in enumerate(problem.workers) if worker.name == name]
        if len(found_workers) != 1:
            raise Exception(f"The amount of workers found for the name {name} is not 1, it's {len(found_workers)}")

        index = found_workers[0][0]
        value = top_solution.values[index]

        return index, value

    for name in names_of_workers:
        var, val = register_for_worker(name)
        result = result.with_fixed_value(var, val)

    return result


def recover_worker_names(worker_names: list[list[str]]):
    pss = [reverse_engineer_ps(names) for names in worker_names]
    destination = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\recovery\pss.npz"

    write_pss_to_file(pss, destination)

def recover_pss():
    worker_names_file = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\recovery\worker_nams.json"
    with open(worker_names_file, "r") as file:
        worker_names = json.load(file)
        recover_worker_names(worker_names)


def adjust_pRef():
    pref_path = r"Experimentation/BT/StaffRosteringProblemCache/pRef_old.npz"
    new_pref_path = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\StaffRosteringProblemCache\pRef_new.npz"
    pRef = PRef.load(pref_path)

    solutions = [FullSolution(row) for row in pRef.full_solution_matrix]
    fitnesses = list(pRef.fitness_array)

    solutions.append(top_solution)
    fitnesses.append(-307.99683917851655)

    new_pRef = PRef.from_full_solutions(solutions, fitnesses, problem.search_space)
    new_pRef.save(new_pref_path)

    again_pRef = PRef.load(new_pref_path)
    print(again_pRef)


adjust_pRef()