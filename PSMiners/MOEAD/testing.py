import warnings


from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PSMiners.MOEAD.Library.aggregation.tchebycheff import Tchebycheff
from PSMiners.MOEAD.Library.algorithm.combinatorial.moead import Moead
from PSMiners.MOEAD.MOEADPSProblem import MOEADPSProblem, PSPolynomialMutation
from utils import announce


def test_moead_on_problem(benchmark_problem: BenchmarkProblem,
                          sample_size: int):
    with announce(f"In test_moead_on_problem, generating a pRef of size {sample_size}"):
        pRef = benchmark_problem.get_reference_population(sample_size)

    moead_problem = MOEADPSProblem(pRef)

    ## the following is copied from https://moead-framework.github.io/framework/html/user_guide.html#overview
    number_of_weight = 10
    # The file is available here : https://github.com/moead-framework/data/blob/master/weights/SOBOL-2objs-10wei.ws
    # Others weights files are available here : https://github.com/moead-framework/data/tree/master/weights
    weight_file = r"C:\Users\gac8\PycharmProjects\PS-PDF\PSMiners\MOEAD\weights\SOBOL-3objs-" + str(number_of_weight) + "wei.ws"
    print("Constructing the moead object")
    ps_evaluations = 1000
    # the following line is required because moead framework still uses np.float when reading the weight files.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    moead = Moead(problem=moead_problem,
                  max_evaluation=ps_evaluations,
                  number_of_weight_neighborhood=2,
                  weight_file=weight_file,
                  aggregation_function=Tchebycheff,
                  genetic_operator=PSPolynomialMutation,
                  number_of_crossover_points=1,   # because I don't want mutation
                  )
    # more options can be found at https://moead-framework.github.io/framework/html/tuto.html#implement-your-own-problem

    with announce("Running moead.run()"):
        population = moead.run()

    print("\nThe population is ")
    for individual in population:
        print(individual)

