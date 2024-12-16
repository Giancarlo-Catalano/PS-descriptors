import numpy as np
from tqdm import tqdm

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.EfficientBTProblem.KnapSackProblem import KnapSackProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.TSP import TSP, BooleanSearchSpaceTSP
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Explanation.PRefManager import PRefManager
from PairExplanation.ExplanationMiner import ExplanationMiner
from utils import announce


def get_unexplained_parts(solution: FullSolution, partial_solutions: list[PS]) -> np.ndarray:
    if len(partial_solutions) == 0:
        return np.ones(shape=len(solution), dtype=bool)
    ps_matrix = np.array([ps.values for ps in partial_solutions])
    return np.all(ps_matrix == STAR, axis=0)


def show_off_problem(problem: BenchmarkProblem,
                     pRef_size: int = 30000,
                     amount_of_pss_to_find: int = 4):
    print(f"The problem is {problem}")

    with announce("Generating the pRef for the GC problem"):
        pRef = PRefManager.generate_pRef(problem=problem,
                                         sample_size=pRef_size,
                                         which_algorithm="uniform GA")

    best_solutions = pRef.get_top_n_solutions(10)
    optima = best_solutions[0]

    explanation_generator = ExplanationMiner(optimisation_problem=problem,
                                             ps_search_budget=5000,
                                             ps_search_population=50,
                                             pRef=pRef,
                                             verbose=True)

    def find_explanation(solution, previously_found_patterns) -> PS:
        unexplained_vars: np.ndarray = get_unexplained_parts(solution, previously_found_patterns)
        print(f"Attempting to find the explanation for {solution}, with the unexplained being {unexplained_vars}")
        return explanation_generator.find_pss(solution,
                                              unexplained_mask=unexplained_vars,
                                              culling_method="biggest",
                                              proportion_used_that_should_be_unexplained=0.7)[0]

    # generate the explanations
    explanations: list[PS] = []
    for _ in tqdm(range(amount_of_pss_to_find)):
        explanation = find_explanation(optima, explanations)
        print(f"The ps {explanation} was found")
        explanations.append(explanation)

    print("The best solution is ")
    print(problem.repr_fs(optima))

    print("The pss are ")
    for ps in explanations:
        print(ps)
        print(problem.repr_ps(ps))
        print("\n\n")


def show_off_graph_colouring():
    connections = []
    connections.extend([(0, 1), (1, 2), (0, 2)])  # a clique of 3
    connections.extend([(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (6, 9)])  # big dipper
    connections.extend([(10, 12), (11, 12), (10, 11), (12, 13), (12, 14), (13, 14)])
    problem = GraphColouring(amount_of_colours=3, amount_of_nodes=15, connections=connections)
    show_off_problem(problem)


def show_off_checkerboard():
    problem = CheckerBoard(5, 5)
    show_off_problem(problem)


def show_off_TSP():
    #original_problem = TSP(cities=[(-1, 2), (0, 2), (2, 2), (-1, 1), (0, 0), (1, -1), (-1, -1)],
    #                      starting_ending_city=(0, 1))
    original_problem = TSP(
        cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8), (7, 9)],
        starting_ending_city=(5, 5))
    problem = BooleanSearchSpaceTSP(original_problem)
    show_off_problem(problem)


def show_off_knapsack():
    items = [
        ("watch", 100, 250.00),
        ("smartphone", 200, 100.00),
        ("cigarettes", 100, 12.00),
        ("razorblades", 100, 10.00),
        ("diluting_juice", 700, 7.00),
        ("notepad", 200, 6.00),
        ("earphones", 70, 5.00),
        ("shirt", 300, 5.00),
        ("milk", 1000, 4.00),
        ("jam", 200, 3.00),
        ("carrots", 300, 2.50),
        ("book", 350, 1.50),
        ("apples", 50, 1.2),
        ("pens", 200, 1.00),
        ("beans", 150, 0.70),
    ]
    problem = KnapSackProblem(items=items,
                              weight_limit=1000)
    show_off_problem(problem)


#show_off_graph_colouring()
