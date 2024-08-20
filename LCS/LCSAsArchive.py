from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from LCS.CustomXCSAlgorithm import CustomXCSAlgorithm
from LCS.XCSProblemTournamenter import XSCProblemTournamenter
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from utils import announce


def check_linkage_metric(ps_evaluator: FastPSEvaluator):
    atomicity_metric = ps_evaluator.atomicity_metric

    nnny = PS(([-1] * 12)+([1]*4))
    ynnn = PS(([1]*4)+([-1] * 12))
    ynny = PS(([1]*4)+([-1] * 8)+([1]*4))

    nnno = PS(([-1] * 12)+([0]*4))
    onnn = PS(([0]*4)+([-1] * 12))
    onno = PS(([0]*4)+([-1] * 8)+([0]*4))

    single_o = PS(([0]*1)+([-1] * 11))


    def print_linkage(ps: PS):
        atomicity = atomicity_metric.get_single_score(ps)
        print(f"The atomicity for {ps} is {atomicity}")

    for ps in [nnny, ynnn, ynny, nnno, onnn, onno, single_o]:
        print_linkage(ps)



def run_LCS_as_archive():
    optimisation_problem = RoyalRoad(5, 4)
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                    sample_size=1000,
                                    which_algorithm="SA uniform")

    ps_evaluator = FastPSEvaluator(pRef)

    xcs_problem = XSCProblemTournamenter(optimisation_problem, pRef = pRef, training_cycles=10000)
    scenario = ScenarioObserver(xcs_problem)
    algorithm = CustomXCSAlgorithm(ps_evaluator)

    algorithm.crossover_probability = 0
    algorithm.deletion_threshold = 10000
    algorithm.discount_factor = 0
    algorithm.do_action_set_subsumption = True
    algorithm.do_ga_subsumption = False
    algorithm.exploration_probability = 0

    model = algorithm.new_model(scenario)

    with announce("Running the model"):
        model.run(scenario, learn=True)

    print("The model is")
    print(model)


run_LCS_as_archive()


