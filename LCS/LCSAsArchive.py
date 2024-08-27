import xcs
from xcs.bitstrings import BitString
from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.FullSolution import FullSolution
from Explanation.PRefManager import PRefManager
from LCS.CustomXCSAlgorithm import CustomXCSAlgorithm
from LCS.XCSProblemTopAndBottom import XCSProblemTopAndBottom
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator
from utils import announce


def set_settings_for_lcs_algorithm(algorithm: xcs.XCSAlgorithm) -> None:
    """Simply sets the settings that are best for my purposes"""
    # play with these settings ad lib. Leaving the defaults seems to work :--)
    # algorithm.crossover_probability = 0
    # algorithm.deletion_threshold = 10000
    # algorithm.discount_factor = 0
    algorithm.do_action_set_subsumption = True
    # algorithm.do_ga_subsumption = False
    # algorithm.exploration_probability = 0
    # algorithm.ga_threshold = 100000
    algorithm.max_population_size = 50
    algorithm.exploration_probability = 0
    # algorithm.minimum_actions = 1


def run_LCS_as_archive(verbose: bool = False):
    # the optimisation problem to be solved

    # optimisation_problem = RoyalRoad(5, 4)
    # optimisation_problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=14, chance_of_connection=0.4)
    optimisation_problem = Trapk(4, 5)
    # optimisation_problem = CheckerBoard(4, 4)

    if isinstance(optimisation_problem, GraphColouring):
        optimisation_problem.view()

    # generates the 'Reference Population', which I call the pRef.
    # the "which algorithm" parameter indicates how this is obtained (there may be multiple sources)
    # e.g uniform SA means that 50% is from random search, 50% is from Simulated Annealing
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                     sample_size=10000,  # these are the Solution evaluations
                                     which_algorithm="uniform SA",
                                     verbose=verbose)

    # Evaluates Linkage and keeps track of PS evaluations used
    ps_evaluator = TMEvaluator(pRef)

    # Shows, alternating, the best and worst solutions to the learner.
    # The best solutions have class 1 and the worst have class 0.
    xcs_problem = XCSProblemTopAndBottom(optimisation_problem,
                                         pRef=pRef,  # where it gets the solutions from
                                         training_cycles=100,  # how many solutions to show (might repeat)
                                         tail_size=30)  # eg show the top n and worst n

    scenario = ScenarioObserver(xcs_problem)

    # my custom XCS algorithm, which just overrides when covering is required, and how it happens
    algorithm = CustomXCSAlgorithm(ps_evaluator,
                                   xcs_problem,
                                   coverage_covering_threshold=0.8,
                                   covering_search_budget=1000,
                                   verbose = verbose)

    set_settings_for_lcs_algorithm(algorithm)

    # This is a custom model, at the time fo this comment it's CustomXCSClassiferSet
    model = algorithm.new_model(scenario)

    with announce("Running the model"):
        model.run(scenario, learn=True)

    print("The model is")
    print(model)

    # We request to evaluate on the best and worst solutions
    solutions_to_evaluate = [xcs_problem.all_solutions[0], xcs_problem.all_solutions[-1]]

    for solution in solutions_to_evaluate:
        result_dict = model.predict(solution)
        rules = result_dict["rules"]

        print(f"Solution: {solution}, "
              f"predicted: {result_dict['prediction']}, "
              f"fitness:{optimisation_problem.fitness_function(solution)}")
        for rule in rules:
            print(rule)


run_LCS_as_archive(verbose=True)
