import random

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, creator, base, tools
from deap.tools import uniform_reference_points, selNSGA3WithMemory, selSPEA2

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace
from Core.TerminationCriteria import TerminationCriteria
from PSMiners.DEAP.CustomCrowdingMechanism import GC_selNSGA3WithMemory


def nsga(toolbox,
         stats,
         mu,
         termination_criteria: TerminationCriteria,
         cxpb,
         mutpb,
         classic3_evaluator: Classic3PSEvaluator,
         verbose=False):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"

    pop = toolbox.population(n=mu)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    iterations = 0
    def should_stop():
        return termination_criteria.met(ps_evaluations = classic3_evaluator.used_evaluations, iterations=iterations)

    while not should_stop():
        pop = list(set(pop))
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, mu)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=iterations, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        iterations +=1

    return pop, logbook



def nsgaiii_pure_functionality(toolbox, mu, ngen, cxpb, mutpb):
    def fill_evaluation_gaps(input_pop):
        for ind in input_pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        return input_pop

    pop = fill_evaluation_gaps(toolbox.population(n=mu))

    # Begin the generational process
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        offspring = fill_evaluation_gaps(offspring)
        pop = toolbox.select(pop + offspring, mu)
    return pop

def geometric_distribution_values_of_ps(search_space: SearchSpace) -> np.ndarray:
    result = PS.empty(search_space)
    chance_of_success = 0.79
    while random.random() < chance_of_success:
        var_index = random.randrange(search_space.amount_of_parameters)
        value = random.randrange(search_space.cardinalities[var_index])
        result = result.with_fixed_value(var_index, value)
    return result


def make_random_deap_individual(search_space: SearchSpace):
    result = geometric_distribution_values_of_ps(search_space)
    return creator.DEAPPSIndividual(result)

def get_toolbox_for_problem(pRef: PRef,
                            classic3_evaluator: Classic3PSEvaluator,
                            uses_experimental_crowding = True,
                            use_spea = False):
    creator.create("FitnessMax", base.Fitness, weights=[1.0, 1.0, 1.0])
    creator.create("DEAPPSIndividual", PS,
                   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    search_space = pRef.search_space


    def make_random_deap_individual():
        result = geometric_distribution_values_of_ps(search_space)
        return creator.DEAPPSIndividual(result)

    toolbox.register("make_random_ps",
                     make_random_deap_individual)
    def evaluate(ps) -> tuple:
        return classic3_evaluator.get_S_MF_A(ps)

    toolbox.register("mate", tools.cxUniform, indpb=1/search_space.amount_of_parameters)
    lower_bounds = [-1 for _ in search_space.cardinalities]
    upper_bounds = [card-1 for card in search_space.cardinalities]
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=1/search_space.amount_of_parameters)

    toolbox.register("evaluate", evaluate)
    toolbox.register("population", tools.initRepeat, list, toolbox.make_random_ps)

    selection_method = None


    ref_points = uniform_reference_points(nobj=3, p=12)
    if not use_spea:
        selection_method = GC_selNSGA3WithMemory(ref_points) if uses_experimental_crowding else selNSGA3WithMemory(ref_points)
    else:
        selection_method = selSPEA2
    toolbox.register("select", selection_method)
    return toolbox

def get_stats_object():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    return stats


def report_in_order_of_last_metric(population,
                                   benchmark_problem: BenchmarkProblem,
                                   limit_to = None):
    population.sort(key=lambda x: x.metric_scores[-1], reverse=True)

    amount_to_show = len(population)
    if limit_to is not None:
        amount_to_show = limit_to
    for ind in population[:amount_to_show]:
        print(benchmark_problem.repr_ps(ind))
        print(f"Has score {ind.metric_scores}\n")


def plot_stats_for_run(logbook,
                       figure_name: str,
                       show_max = False,
                       show_mean = True,
                       ):
    generations = logbook.select("gen")
    metric_labels = ["Simplicity", "MeanFitness", "Atomicity"]
    num_variables = len(metric_labels)


    avg_matrix = np.array([logbook[generation]["avg"] for generation in generations])
    max_matrix = np.array([logbook[generation]["max"] for generation in generations])

    # Create a new figure with subplots
    fig, axs = plt.subplots(1, num_variables, figsize=(12, 6))  # 1 row, `num_variables` columns

    # Loop through each variable to create a subplot
    for metric_index, metric_label in enumerate(metric_labels):

        if show_mean:
            axs[metric_index].plot(generations, avg_matrix[:, metric_index], label='Average', linestyle='-', marker='o')
        if show_max:
            axs[metric_index].plot(generations, max_matrix[:, metric_index], label='Maximum', linestyle='--', marker='x')

        axs[metric_index].set_xlabel('Generation')
        axs[metric_index].set_ylabel('Value')
        axs[metric_index].set_title(metric_label)
        axs[metric_index].legend()  # Add legend to each subplot

    # Adjust layout to ensure plots don't overlap
    plt.tight_layout()

    # Display the plots
    plt.savefig(figure_name)
    plt.close(fig)






