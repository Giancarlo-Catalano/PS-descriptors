import xcs
from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from LCS.XCSComponents.SolutionDifferenceAlgorithm import SolutionDifferenceAlgorithm
from LCS.XCSComponents.SolutionDifferenceScenario import SolutionDifferenceScenario
from LCS.PSEvaluator import GeneralPSEvaluator
from utils import announce


def set_settings_for_lcs_algorithm(algorithm: xcs.XCSAlgorithm) -> None:
    """Simply sets the settings that are best for my purposes"""
    # play with these settings ad lib. Leaving the defaults seems to work :--)
    algorithm.crossover_probability = 0
    algorithm.deletion_threshold = 10  #  minimum age before a rule can be pruned away
    # algorithm.discount_factor = 0
    algorithm.do_action_set_subsumption = True
    # algorithm.do_ga_subsumption = True
    # algorithm.exploration_probability = 0
    # algorithm.ga_threshold = 100000
    algorithm.max_population_size = 50
    # algorithm.exploration_probability = 0
    # algorithm.minimum_actions = 1
    algorithm.subsumption_threshold = 50  # minimum age before a rule can subsume another

    algorithm.allow_ga_reproduction = False


def run_solution_difference_LCS(verbose: bool = False):
    # the optimisation problem to be solved

    # optimisation_problem = RoyalRoad(10, 4)
    # optimisation_problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.4)
    # optimisation_problem = Trapk(4, 5)
    # optimisation_problem = CheckerBoard(4, 4)
    # optimisation_problem = EfficientBTProblem.from_default_files()

    if isinstance(optimisation_problem, GraphColouring):
        optimisation_problem.view()

    # generates the 'Reference Population', which I call the pRef.
    # the "which algorithm" parameter indicates how this is obtained (there may be multiple sources)
    # e.g uniform SA means that 50% is from random search, 50% is from Simulated Annealing
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                     sample_size=10000,  # these are the Solution evaluations
                                     which_algorithm="uniform GA",
                                     verbose=verbose)

    pRef = PRef.unique(pRef)
    print(f"After removing duplicate entries, pRef has {pRef.sample_size} samples")

    # Evaluates Linkage and keeps track of PS evaluations used
    ps_evaluator = GeneralPSEvaluator(pRef)

    # Shows, alternating, the best and worst solutions to the learner.
    # The best solutions have class 1 and the worst have class 0.
    xcs_problem = SolutionDifferenceScenario(original_problem=optimisation_problem,
                                             pRef=pRef,  # where it gets the solutions from
                                             training_cycles=10000,  # how many solutions to show (might repeat)
                                             verbose=verbose)

    scenario = ScenarioObserver(xcs_problem)

    # my custom XCS algorithm, which just overrides when covering is required, and how it happens
    algorithm = SolutionDifferenceAlgorithm(ps_evaluator=ps_evaluator,
                                            xcs_problem = xcs_problem,
                                            covering_search_budget=1000,
                                            covering_population_size=50,
                                            verbose=verbose)

    set_settings_for_lcs_algorithm(algorithm)

    # This is a custom model, at the time fo this comment it's CustomXCSClassiferSet
    model = algorithm.new_model(scenario)
    model.verbose = True

    with announce("Running the model"):
        model.run(scenario, learn=True)

    print("The model is")

    model_rules = [item[True] for item in model._population.values()]
    model_rules.sort(key=lambda x: x.average_reward, reverse=True)
    for rule in model_rules:
        assert(isinstance(rule, xcs.ClassifierRule))
        print(optimisation_problem.repr_ps(rule.condition))
        print(rule)

    print("End of run")

    # # We request to evaluate on the best and worst solutions
    # solutions_to_evaluate = [xcs_problem.all_solutions[0], xcs_problem.all_solutions[-1]]
    #
    # for solution in solutions_to_evaluate:
    #     result_dict = model.predict(solution)
    #     rules = result_dict["rules"]
    #
    #     print(f"Solution: {solution}, "
    #           f"predicted: {result_dict['prediction']}, "
    #           f"fitness:{optimisation_problem.fitness_function(solution)}")
    #     for rule in rules:
    #         print(rule)
    # Submitting to the uni computer


run_solution_difference_LCS(verbose=True)
