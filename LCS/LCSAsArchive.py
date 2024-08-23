import xcs
from xcs.bitstrings import BitString
from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.FullSolution import FullSolution
from Explanation.PRefManager import PRefManager
from LCS.CustomXCSAlgorithm import CustomXCSAlgorithm
from LCS.XCSProblemTopAndBottom import XCSProblemTopAndBottom
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator
from utils import announce


def use_model_for_prediction(model: xcs.ClassifierSet, solution: FullSolution) -> dict:
    as_input = BitString(solution.values)
    match_set = model.match(as_input)
    selected_action = match_set.select_action()
    rules = list(match_set[selected_action])

    def get_rule_quality(rule):
        return rule.prediction_weight

    rules.sort(key=get_rule_quality)

    result = {"selected_action": selected_action,
              "rules": rules}

    return result



def run_LCS_as_archive():
    # the optimisation problem to be solved
    optimisation_problem = RoyalRoad(4, 4) #GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.4)
    # optimisation_problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=14, chance_of_connection=0.4)
    # optimisation_problem = Trapk(4, 5)
    # optimisation_problem = CheckerBoard(4, 4)

    if isinstance(optimisation_problem, GraphColouring):
        optimisation_problem.view()

    # generates the 'Reference Population', which I call the pRef.
    # the "which algorithm" parameter indicates how this is obtained (there may be multiple sources)
    # e.g uniform SA means that 50% is from random search, 50% is from Simulated Annealing
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                    sample_size=10000,  # these are the Solution evaluations
                                    which_algorithm="uniform GA")

    # Evaluates Linkage and keeps track of PS evaluations used
    ps_evaluator = TMEvaluator(pRef)

    # shows, alternating, the best and worst solutions. The best solutions have class 1 and the worst have class 0.
    xcs_problem = XCSProblemTopAndBottom(optimisation_problem,
                                         pRef = pRef,          # where it gets the solutions from
                                         training_cycles=3000, # how many solutions to show (might repeat)
                                         tail_size = 1000)     # eg show the top n and worst n

    scenario = ScenarioObserver(xcs_problem)

    # my custom XCS algorithm, which just overrides when covering is required, and how it happens
    algorithm = CustomXCSAlgorithm(ps_evaluator, xcs_problem, verbose=True)

    # play with these settings ad libidem. Leaving the defaults seems to work :--)
    # algorithm.crossover_probability = 0
    # algorithm.deletion_threshold = 10000
    # algorithm.discount_factor = 0
    # algorithm.do_action_set_subsumption = True
    # algorithm.do_ga_subsumption = False
    # algorithm.exploration_probability = 0
    # algorithm.ga_threshold = 100000
    # algorithm.max_population_size = 50
    # algorithm.exploration_probability = 0
    # algorithm.minimum_actions = 1

    # This is a custom model, at the time fo this comment it's CustomXCSClassiferSet
    model = algorithm.new_model(scenario)

    with announce("Running the model"):
        model.run(scenario, learn=True)

    print("The model is")
    print(model)

    # We requst to evaluate on the best and worst solutions
    solutions_to_evaluate = [xcs_problem.all_solutions[0], xcs_problem.all_solutions[-1]]


    for solution in solutions_to_evaluate:
        as_input = BitString(solution.values)
        match_set = model.match(as_input)

        result_dict = use_model_for_prediction(model, solution)
        rules = result_dict["rules"]

        print(f"Solution: {solution}, "
              f"predicted: {match_set.select_action()}, "
              f"fitness:{optimisation_problem.fitness_function(solution)}")
        for rule in rules:
            print(rule)

run_LCS_as_archive()


